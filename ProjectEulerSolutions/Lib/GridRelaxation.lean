import ProjectEulerStatements.P83
import Mathlib.Data.Int.Basic

/-! Individual synchronous relaxations in the four-direction grid reference. -/
namespace ProjectEulerSolutions.GridRelaxation
open ProjectEulerStatements.P83

def cell (matrix distances : List (List Int)) (r c : Int) : Int :=
  let current := getD2 distances r c
  let adjacent := neighbors r c matrix.length (matrix.headD []).length
  let least := adjacent.foldl (fun best pair => min best (getD2 distances pair.1 pair.2)) current
  min current (getD2 matrix r c + least)

theorem relaxStep_eq (matrix distances : List (List Int)) :
    relaxStep matrix distances = (List.range matrix.length).map (fun (r : Nat) =>
      (List.range (matrix.headD []).length).map (fun (c : Nat) => cell matrix distances r c)) := rfl

theorem cell_le (matrix distances : List (List Int)) (r c : Int) :
    cell matrix distances r c ≤ getD2 distances r c := by
  unfold cell
  exact Int.min_le_left _ _

end ProjectEulerSolutions.GridRelaxation
