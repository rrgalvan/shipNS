all: rectang_circ_hole.xdmf

rectang_circ_hole.msh: rectang_circ_hole.geo
 $< -format msh2

rectang_circ_hole.xdmf: rectang_circ_hole.msh
	meshio convert $< $@
