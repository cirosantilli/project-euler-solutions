import ProjectEulerSolutions.Lib.MonopolyIntegerWeights

namespace ProjectEulerSolutions.Monopoly
open scoped BigOperators
set_option Elab.async false

def squareWeight (values : Fin 120 → Int) (square : Fin 40) : Int :=
  ∑ doubles : Fin 3, values (stateIndex (square, doubles))

theorem IntegerWeights.square_proportional {sides : Nat} (positive : 0 < sides) {left right : Fin 120 → Int}
    (validLeft : IntegerWeights sides left) (validRight : IntegerWeights sides right) (square : Fin 40) :
    squareWeight left square * (∑ index, right index) = squareWeight right square * (∑ index, left index) := by
  simp only [squareWeight, Finset.sum_mul]
  exact Finset.sum_congr rfl (fun doubles _ => validLeft.proportional positive validRight (stateIndex (square, doubles)))

theorem IntegerWeights.square_lt_iff {sides : Nat} (positive : 0 < sides) {left right : Fin 120 → Int}
    (validLeft : IntegerWeights sides left) (validRight : IntegerWeights sides right) (first second : Fin 40) :
    squareWeight left first < squareWeight left second ↔ squareWeight right first < squareWeight right second := by
  calc
    _ ↔ squareWeight left first * (∑ index, right index) < squareWeight left second * (∑ index, right index) :=
      (mul_lt_mul_iff_left₀ validRight.1).symm
    _ ↔ squareWeight right first * (∑ index, left index) < squareWeight right second * (∑ index, left index) := by
      rw [validLeft.square_proportional positive validRight first, validLeft.square_proportional positive validRight second]
    _ ↔ _ := mul_lt_mul_iff_left₀ validLeft.1

theorem IntegerWeights.square_le_iff {sides : Nat} (positive : 0 < sides) {left right : Fin 120 → Int}
    (validLeft : IntegerWeights sides left) (validRight : IntegerWeights sides right) (first second : Fin 40) :
    squareWeight left first ≤ squareWeight left second ↔ squareWeight right first ≤ squareWeight right second := by
  simpa only [not_lt] using not_congr (validLeft.square_lt_iff positive validRight second first)

theorem IntegerWeights.square_eq_iff {sides : Nat} (positive : 0 < sides) {left right : Fin 120 → Int}
    (validLeft : IntegerWeights sides left) (validRight : IntegerWeights sides right) (first second : Fin 40) :
    squareWeight left first = squareWeight left second ↔ squareWeight right first = squareWeight right second := by
  constructor
  · intro equal
    exact le_antisymm ((validLeft.square_le_iff positive validRight first second).mp equal.le)
      ((validLeft.square_le_iff positive validRight second first).mp equal.ge)
  · intro equal
    exact le_antisymm ((validLeft.square_le_iff positive validRight first second).mpr equal.le)
      ((validLeft.square_le_iff positive validRight second first).mpr equal.ge)

def ranksBefore (values : Fin 120 → Int) (first second : Fin 40) : Prop :=
  squareWeight values second < squareWeight values first ∨
    (squareWeight values first = squareWeight values second ∧ first.val < second.val)

/-- Both descending probabilities and exact index tie breaks are independent of the integer normalization. -/
theorem IntegerWeights.ranking_iff {sides : Nat} (positive : 0 < sides) {left right : Fin 120 → Int}
    (validLeft : IntegerWeights sides left) (validRight : IntegerWeights sides right) (first second : Fin 40) :
    ranksBefore left first second ↔ ranksBefore right first second := by
  simp only [ranksBefore, validLeft.square_lt_iff positive validRight, validLeft.square_eq_iff positive validRight]

end ProjectEulerSolutions.Monopoly
