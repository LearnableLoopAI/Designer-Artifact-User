# Designer-Artifact-User
Multi-agent active inference with PyMDP

# **Designer-Artifact-User multi-agent system**

# 1 Summary of entity interactions
The following diagram is a representation of the interactions between the entities in this project. 

![](./designer-artifact-user3.png)

The Designer entity's agent has symbols:

- $a^{\mathrm{DsgArt}}$ is the action from the Designer entity's agent to the Artifact entity's environment segment
- $u^{\mathrm{ArtDsg}}$ is the inferred control states of the Artifact entity's environment segment
- $\theta^{\mathrm{ArtDsg}}$ is the inferred parameters of the Artifact entity's environment segment
- $s^{\mathrm{ArtDsg}}$ is the inferred state of the Artifact entity's environment segment
- $\theta^{\mathrm{UsrDsg}}$ is the inferred parameters of the User entity's environment segment
- $s^{\mathrm{UsrDsg}}$ is the inferred state of the User entity's environment segment

The Designer entity's environment has symbols:

- from Artifact's environment segment:
    - $y^{\mathrm{ArtDsg}}$ is the observation from the Artifact entity's environment segment
    - $\breve\theta^{\mathrm{ArtDsg}}$ is the true parameters of the Artifact entity
    - $\breve s^{\mathrm{ArtDsg}}$ is the true state of the Artifact entity
    - $W^{\mathrm{Art}}$ is the exogenous information impacting the Artifact entity
- from User's environment segment:
    - $y^{\mathrm{UsrDsg}}$ is the observation from the User entity's environment segment
    - $\breve\theta^{\mathrm{UsrDsg}}$ is the true parameters of the User entity
    - $\breve s^{\mathrm{UsrDsg}}$ is the true state of the User entity
