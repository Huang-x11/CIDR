import torch
import torch.nn as nn

from core.abmil import ABMIL
from core.clam import CLAM_MB, CLAM_SB
from core.dsmil import DSMIL
from core.dtfd import DTFD
from core.maxmil import MaxMIL
from core.meanmil import MeanMIL
from core.transmil import TransMIL


def build_mil_classifier(backbone, input_dim):
    """Build a bag-level classifier with the same interface as the legacy code."""
    if backbone == "MeanMIL":
        return MeanMIL(input_dim)
    if backbone == "MaxMIL":
        return MaxMIL(input_dim)
    if backbone == "ABMIL":
        return ABMIL(input_dim)
    if backbone == "TransMIL":
        return TransMIL(input_dim)
    if backbone == "CLAM-SB":
        return CLAM_SB(input_dim)
    if backbone == "CLAM-MB":
        return CLAM_MB(input_dim)
    if backbone == "DSMIL":
        return DSMIL(input_dim)
    if backbone == "DTFD-MaxMinS":
        return DTFD(input_dim, distill="MaxMinS")
    if backbone == "DTFD-AFS":
        return DTFD(input_dim, distill="AFS")
    if backbone == "DTFD-MaxS":
        return DTFD(input_dim, distill="MaxS")
    raise ValueError("Unknown MIL backbone: {}".format(backbone))


class Encoder(nn.Module):
    """Deterministically split each frozen PFM feature into S and Z."""

    def __init__(self, input_dim, hidden_dim, key_concepts, confuse_concepts, concept_dim):
        super().__init__()
        self.key_concepts = key_concepts
        self.confuse_concepts = confuse_concepts
        self.concept_dim = concept_dim

        # S: label-causative / task-relevant representation.
        self.s_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, key_concepts * concept_dim),
        )

        # Z: label-non-causative residual representation.
        self.z_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, confuse_concepts * concept_dim),
        )

    def forward(self, x):
        patch_count = x.shape[0]
        s = self.s_encoder(x).reshape(patch_count, self.key_concepts, self.concept_dim)
        z = self.z_encoder(x).reshape(patch_count, self.confuse_concepts, self.concept_dim)
        return s, z


class Decoder(nn.Module):
    """Reconstruct the original PFM feature from the joint (S, Z) representation."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, latent):
        return self.decoder(latent)


class HSIC(nn.Module):
    """Mean normalized RBF-HSIC of [one current patch + all reference patches].

    Every current patch is an anchor; none is subsampled. Reference bandwidths
    are detached and shared within the batch. All kernel values retain gradients.
    """

    def __init__(self, chunk_size=512, eps=1e-12, bandwidth_floor=1e-8):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("HSIC chunk_size must be positive.")
        self.chunk_size = chunk_size
        self.eps = eps
        self.bandwidth_floor = bandwidth_floor

    @staticmethod
    def squared_distances(x, y):
        return (x.square().sum(1, keepdim=True) + y.square().sum(1).unsqueeze(0) - 2 * x @ y.t()).clamp_min(0)

    def reference_kernel(self, x):
        distances = self.squared_distances(x, x)
        indices = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
        bandwidth = distances[indices[0], indices[1]].detach().median().clamp_min(self.bandwidth_floor)
        kernel = torch.exp(-distances / (2 * bandwidth))
        # RBF self similarities are exactly one, including in finite precision.
        kernel = kernel - torch.diag_embed(kernel.diagonal()) + torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
        return kernel, bandwidth

    @staticmethod
    def reference_stats(kernel):
        row_mean = kernel.mean(1)
        grand_mean = row_mean.mean()
        centered = kernel - row_mean[:, None] - row_mean[None, :] + grand_mean
        return centered, row_mean, grand_mean

    @staticmethod
    def anchor_stats(cross_kernel, row_mean, grand_mean):
        delta = cross_kernel - row_mean
        delta = delta - delta.mean(1, keepdim=True)
        contrast = 1 - 2 * cross_kernel.mean(1) + grand_mean
        return delta, contrast

    def forward(self, s, z, reference_s, reference_z):
        if s.ndim != 2 or z.ndim != 2 or reference_s.ndim != 2 or reference_z.ndim != 2:
            raise ValueError("HSIC expects flattened [patches, features] tensors.")
        if s.shape[0] == 0 or s.shape[0] != z.shape[0] or reference_s.shape[0] != reference_z.shape[0]:
            raise ValueError("HSIC requires nonempty, aligned S/Z pairs.")
        if reference_s.shape[0] < 2:
            raise ValueError("HSIC needs at least two external training slides.")
        # Small centered kernel statistics use float64 to avoid cancellation.
        # Conversion preserves gradients back to the float32 Encoder.
        original_dtype = s.dtype
        s, z, reference_s, reference_z = s.double(), z.double(), reference_s.double(), reference_z.double()
        a, bandwidth_s = self.reference_kernel(reference_s)
        b, bandwidth_z = self.reference_kernel(reference_z)
        ac, ar, am = self.reference_stats(a)
        bc, br, bm = self.reference_stats(b)
        ab, aa, bb = (ac * bc).sum(), ac.square().sum(), bc.square().sum()
        ratio = reference_s.shape[0] / (reference_s.shape[0] + 1)
        total = s.new_zeros(())

        for start in range(0, s.shape[0], self.chunk_size):
            u = torch.exp(-self.squared_distances(s[start:start + self.chunk_size], reference_s) / (2 * bandwidth_s))
            v = torch.exp(-self.squared_distances(z[start:start + self.chunk_size], reference_z) / (2 * bandwidth_z))
            du, cu = self.anchor_stats(u, ar, am)
            dv, cv = self.anchor_stats(v, br, bm)
            # Exact centered inner products for each augmented (k+1)-sample set.
            numerator = ab + 2 * ratio * (du * dv).sum(1) + ratio ** 2 * cu * cv
            norm_s = aa + 2 * ratio * du.square().sum(1) + ratio ** 2 * cu.square()
            norm_z = bb + 2 * ratio * dv.square().sum(1) + ratio ** 2 * cv.square()
            denominator = (norm_s * norm_z).clamp_min(self.eps ** 2).sqrt()
            total = total + (numerator / (denominator + self.eps)).sum()

        # Sum over anchors, not an unweighted mean over differently sized chunks.
        return (total / s.shape[0]).to(original_dtype)


class CDG(nn.Module):
    """Task-level S/Z model with S -> Y and (S, Z) -> X assumptions.
    """

    def __init__(self, backbone, input_dim=1024, hidden_dim=256, key_concepts=4, confuse_concepts=12, concept_dim=16, hsic_chunk_size=512, lambda_cls=1.0, lambda_rec=1.0, lambda_kl=1.0, lambda_hsic=1e-3):
        super().__init__()
        self.key_concepts = key_concepts
        self.confuse_concepts = confuse_concepts
        self.concept_dim = concept_dim

        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec
        self.lambda_kl = lambda_kl
        self.lambda_hsic = lambda_hsic

        latent_dim = (key_concepts + confuse_concepts) * concept_dim
        s_dim = key_concepts * concept_dim

        self.encoder = Encoder(input_dim, hidden_dim, key_concepts, confuse_concepts, concept_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)
        self.classifier = build_mil_classifier(backbone, s_dim)
        self.hsic = HSIC(chunk_size=hsic_chunk_size)
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def kl_normal(mu, logvar):
        # This intentionally preserves the deterministic legacy KL definition.
        return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

    def compute_losses(self, x, y, bag_sizes, reference_x=None):
        s, z = self.encoder(x)
        s_flat = s.flatten(start_dim=1)
        z_flat = z.flatten(start_dim=1)

        cls_probs, loss_cls = self.classifier(s_flat, y, bag_sizes)

        latent = torch.cat([s, z], dim=1)
        latent_flat = latent.flatten(start_dim=1)
        recon_x = self.decoder(latent_flat)
        loss_rec = self.mse_loss(recon_x, x)

        logvar = torch.zeros_like(latent_flat)
        loss_kl = self.kl_normal(latent_flat, logvar)

        loss_hsic = latent_flat.new_zeros(())
        if reference_x is not None and self.lambda_hsic != 0:
            if len(bag_sizes) != 1:
                raise ValueError("Cross-slide anchor HSIC requires one current WSI per batch.")
            reference_s, reference_z = self.encoder(reference_x)
            loss_hsic = self.hsic(s_flat, z_flat, reference_s.flatten(start_dim=1), reference_z.flatten(start_dim=1))

        loss = self.lambda_cls * loss_cls + self.lambda_rec * loss_rec + self.lambda_kl * loss_kl + self.lambda_hsic * loss_hsic

        return {
            "loss": loss,
            "probs": cls_probs,
            "loss_cls": loss_cls,
            "loss_rec": loss_rec,
            "loss_kl": loss_kl,
            "loss_hsic": loss_hsic,
            "s": s,
            "z": z,
            "recon_x": recon_x,
        }

    def forward(self, x, y, bag_sizes, reference_x=None):
        outputs = self.compute_losses(x, y, bag_sizes, reference_x)
        return outputs["loss"], outputs["probs"]

