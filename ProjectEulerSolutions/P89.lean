import ProjectEulerStatements.P89
import ProjectEulerSolutions.Lib.RomanNumerals

namespace ProjectEulerSolutions.P89

def valueOf (c : Char) : Nat :=
  if c = 'I' then 1 else if c = 'V' then 5 else if c = 'X' then 10
  else if c = 'L' then 50 else if c = 'C' then 100 else if c = 'D' then 500
  else if c = 'M' then 1000 else 0

def parseAcc : List Char → Nat → Nat
  | [], total => total
  | [x], total => total + valueOf x
  | x :: y :: ys, total =>
      if valueOf x < valueOf y then parseAcc ys (total + (valueOf y - valueOf x))
      else parseAcc (y :: ys) (total + valueOf x)

def romanToInt (s : String) : Nat := parseAcc s.toList 0

abbrev minimalTable : List (Nat × String) := [
  (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
  (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]

/-- Quotient/remainder token emission, as in the Python solver. -/
def intToMinRoman (x : Nat) : String :=
  String.ofList (RomanNumerals.encode (minimalTable.map (fun p => (p.1, p.2.toList))) x)

def totalCharactersSaved (lines : List String) : Nat :=
  lines.foldl (fun acc s => acc + (s.length - (intToMinRoman (romanToInt s)).length)) 0

def solve (lines : List String) : Nat := totalCharactersSaved lines
end ProjectEulerSolutions.P89
