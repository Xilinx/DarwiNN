
all: darwinn.sif

darwinn.sif: darwinn.def
	$(SUDO) singularity build $@ $<

singularity-cli_latest.sif:
	singularity pull shub://singularityhub/singularity-cli:latest

darwinn.def: Dockerfile singularity-cli_latest.sif
	singularity exec singularity-cli_latest.sif spython -i docker recipe $< > $@


