import ProjectEulerStatements.P98
namespace ProjectEulerSolutions.P98

/-- Canonical digit/letter signatures reject incompatible square assignments before substitution. -/
def solve (words : List String) : Int := ProjectEulerSolutions.AnagramSearch.solve words

def parseWords (text : String) : List String :=
  (text.replace "\"" "").splitOn "," |>.map (fun word => word.trimAscii.toString)
end ProjectEulerSolutions.P98
