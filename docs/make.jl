using Documenter, JuLIP

makedocs(
      modules = [JuLIP],
      clean = false,
      format   = Documenter.Formats.HTML,
      sitename = "JuLIP.jl",
      pages = [
         "Home" => "index.md",
         "Implementation Notes" => "ImplementationNotes.md",
         "Temporary Notes" => "tempnotes.md"
      ]
   )

# deploydocs(
#     julia = "nightly",
#     repo = "github.com/JuliaDocs/Documenter.jl.git",
#     target = "build",
#     deps = nothing,
#     make = nothing,
# )
