
function h0(str)
   dashes = "≡"^(length(str)+4)
   printstyled(dashes, color=:magenta); println()
   printstyled("  "*str*"  ", bold=true, color=:magenta); println()
   printstyled(dashes, color=:magenta); println()
end

function h1(str)
   dashes = "="^(length(str)+2)
   printstyled(dashes, color=:magenta); println()
   printstyled(" " * str * " ", bold=true, color=:magenta); println()
   printstyled(dashes, color=:magenta); println()
end

function h2(str)
   dashes = "-"^length(str)
   printstyled(dashes, color=:magenta); println()
   printstyled(str, bold=true, color=:magenta); println()
   printstyled(dashes, color=:magenta); println()
end

h3(str) = (printstyled(str, bold=true, color=:magenta); println())
