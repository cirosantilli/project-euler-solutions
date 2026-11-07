import ProjectEulerStatements.Lib.ChampernowneBlocks
namespace ProjectEulerSolutions.P40

/-- Locate the digit by skipping complete blocks of equal-width decimal numbers. -/
def digitAt (position : Nat) : Nat :=
  ProjectEulerStatements.ChampernowneBlocks.blockDigit 1 position

def solve (positions : List Nat) : Nat :=
  positions.foldl (fun product position => product * digitAt position) 1
end ProjectEulerSolutions.P40
