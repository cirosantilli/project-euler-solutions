import ProjectEulerStatements.P29
namespace ProjectEulerSolutions.P29

/-- Enumerate powers and retain each value once, as in the Python set comprehension. -/
def solve (maxA maxB : Nat) : Nat :=
  (((List.range (maxA - 1)).flatMap fun a =>
    (List.range (maxB - 1)).map fun b => (a + 2) ^ (b + 2)).eraseDups).length
end ProjectEulerSolutions.P29
