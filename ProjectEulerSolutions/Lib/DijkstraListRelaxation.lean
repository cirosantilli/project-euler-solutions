import ProjectEulerSolutions.Lib.DijkstraArrayRelaxation

/-! A list buffer with an unconditional comparison represents guarded array
relaxation when settled vertices cannot be improved. -/
namespace ProjectEulerSolutions.DijkstraArrayRelaxation
open ProjectEulerSolutions.P83

theorem list_relax_agreement (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (values : List Int) (cell : Int × Int)
    (inside : cell.1.toNat * width + cell.2.toNat < values.length)
    (settled : visited[cell.1.toNat * width + cell.2.toNat]! = true →
      values[cell.1.toNat * width + cell.2.toNat] ≤ distance + ProjectEulerStatements.P83.getD2 matrix cell.1 cell.2) :
    (values.set (cell.1.toNat * width + cell.2.toNat)
      (min values[cell.1.toNat * width + cell.2.toNat]
        (distance + ProjectEulerStatements.P83.getD2 matrix cell.1 cell.2))).toArray =
      relax matrix width visited distance values.toArray cell := by
  have read : values.toArray[cell.1.toNat * width + cell.2.toNat]! = values[cell.1.toNat * width + cell.2.toNat] := by
    simp [getElem!_pos, inside]
  unfold relax
  simp only [read, Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq]
  by_cases smaller : distance + ProjectEulerStatements.P83.getD2 matrix cell.1 cell.2 <
      values[cell.1.toNat * width + cell.2.toNat]
  · have waiting : visited[cell.1.toNat * width + cell.2.toNat]! = false := by
      cases flag : visited[cell.1.toNat * width + cell.2.toNat]! with
      | false => rfl
      | true => have bound := settled flag; omega
    rw [if_pos ⟨waiting, smaller⟩, Int.min_eq_right (by omega)]
    simp only [Array.set!_eq_setIfInBounds, List.setIfInBounds_toArray]
  · rw [if_neg (by omega), Int.min_eq_left (by omega), List.set_getElem_self inside]

end ProjectEulerSolutions.DijkstraArrayRelaxation
