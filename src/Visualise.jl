
module Visualise

using PyCall
using JuLIP.ASE: ASEAtoms, write

@pyimport IPython.display as ipydisp
@pyimport imolecule

"""
# Varargs (copied from imolecule)

* `size`: Starting dimensions of visualization, in pixels, e.g., `(500,500)`
* `drawing_type`: Specifies the molecular representation. Can be 'ball and
    stick', "wireframe", or 'space filling'.
* `camera_type`: Can be 'perspective' or 'orthographic'.
* `shader`: Specifies shading algorithm to use. Can be 'toon', 'basic',
    'phong', or 'lambert'.
* `display_html`: If True (default), embed the html in a IPython display.
    If False, return the html as a string.
* `element_properites`: A dictionary providing color and radius information
    for custom elements or overriding the defaults in imolecule.js
* `show_save`: If True, displays a save icon for rendering molecule as an
    image.
"""
function Base.view(at::ASEAtoms;
         camera_type="perspective", size=(500,500), display_html=false, varargs...)
    fn = "$(tempname()).xyz"
    write(fn, at)
    out = ipydisp.HTML(
         imolecule.draw(fn, format="xyz", camera_type=camera_type,
                              size=size, display_html=false, varargs...)
                     )
    rm(fn)
    return out
end



end
