using Tutorials
using Documenter

makedocs(;
    modules=[Tutorials],
    authors="SciQuant",
    repo="https://github.com/rvignolo/Tutorials.jl/blob/{commit}{path}#L{line}",
    sitename="Tutorials.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
