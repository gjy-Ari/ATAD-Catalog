=======================================================================
Description of the Asteroid Type, Albedo, and Diameter Catalog (ATAD)
=======================================================================
This catalog contains taxonomic types, visible-band geometric albedo and effective diameter of 188,742 asteroids.

Types are sourced from SsODNet (Berthier et al., 2023) and the AGAI catalog (Ge et al., 2025). These have been unified into six classes: A, C, D, S, V, and X (Ge et al., 2024). 

The albedo and diameter are simultaneously estimated by a deep learning model (AadNet2) using the following input parameters:
semimajor axis, eccentricity, inclination, perihelion distance, aphelion distance, orbital period, semilatus rectum, absolute magnitude, type, type quality, and initial albedo.

=======================================================================
Attribute fields in the ATAD catalog
=======================================================================
Number: MPC asteroid number;
Type: Taxonomic classification;
Type_method: Indicates the classification data, spectral bands, or whether it is predicted by machine learning;
Type_confidence: Confidence value of the machine learning classification (Ge et al., 2025) ;
Albedo: Geometric albedo predicted by the AadNet2 model;
Albedo uncertainty: Uncertainty of the predicted albedo;
Diameter (km): Effective diameter predicted by the AadNet2 model;
Diameter uncertainty (km): Uncertainty of the predicted effective diameter.

=======================================================================
References
=======================================================================
Berthier J, Carry B, Mahlke M, et al. 2023. SsODNet: Solar system Open Database Network[J]. Astronomy & Astrophysics, 671: A151.
Ge J., Zhang X., Li J., et al. 2024. Asteroid material classification based on multi-parameter constraints using artificial intelligence[J]. Astronomy & Astrophysics, 692, A100.
Ge J, Zhang X, Li J, et al. 2025. Asteroid Types, Albedos, and Diameters Catalog from Gaia DR3: Intelligent Inversion Results via Multisource Information Fusion[J]. The Astrophysical Journal Supplement Series, 280(1): 17.
