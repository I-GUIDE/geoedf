# GeoEDF Connectors and Processors
Each folder in this repository is the set of files required to build a GeoEDF connector or processor. The recipe.hpccm file 
in each folder can be used to generate a Singularity or Dockerfile container recipe, which can be used to build a container image 
to place on the desired HPC system. Connector and processor execution uses CyberGIS-Compute. The necessary manifest for each such 
CyberGIS-Compute model can be found in their respective repositories.
