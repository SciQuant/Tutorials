using Tutorials
using Test

function get_title(files,filename)
  for (title,fn) in files
    if fn == filename
      return title
    end
  end
  error("File $filename not found!")
end

@show ARGS

if (length(ARGS) != 0)
  files = [get_title(Tutorials.files,filename)=>filename for filename in ARGS]
else
  files = Tutorials.files
end

for (title,filename) in files
    # Create temporal modules to isolate and protect test scopes
    tmpdir=mktempdir(;cleanup=true)
    filename_wo_extension=split(filename,".")[1]
    tmpmod = filename_wo_extension
    tmpfile = joinpath(tmpdir,tmpmod)
    isfile(tmpfile) && error("File $tmpfile already exists!")
    testpath = joinpath(@__DIR__,"../src", filename)
    open(tmpfile,"w") do f
      println(f, "# This file is automatically generated")
      println(f, "# Do not edit")
      println(f)
      println(f, "module $tmpmod include(\"$testpath\") end")
    end
    @time @testset "$title" begin include(tmpfile) end
end

# module fsi_tutorial
# using Test
# @time @testset "fsi_tutorial" begin include("../src/fsi_tutorial.jl") end
# end # module