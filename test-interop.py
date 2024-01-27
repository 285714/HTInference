print("loading")
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Base
print("---->")
print(Base.sind(90))
print(Base.sind(0))

