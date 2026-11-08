import solvers.«84.py.lean.d».OutcomeLoop
import ProjectEulerSolutions.Lib.DiceSums

namespace solvers.«84».outcomes
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

theorem count_nat (sides move : Nat) (lower : 2 ≤ move) (upper : move ≤ 2 * sides) :
    count sides move = (min (move - 1) (2 * sides + 1 - move) : Nat) := by
  unfold count
  omega

theorem double_nat (move : Nat) : double move = ((if move % 2 = 0 then 1 else 0) : Nat) := by
  have parity : (move : Int) % 2 = 0 ↔ move % 2 = 0 := by omega
  by_cases even : move % 2 = 0 <;> simp [double, parity, even]

theorem count_ge_double (sides move : Nat) (lower : 2 ≤ move) (upper : move ≤ 2 * sides) :
    double move ≤ count sides move := by
  rw [count_nat sides move lower upper, double_nat]
  split <;> omega

theorem per_move_weight (sides move : Int) (flag : Bool) (enough : double move ≤ count sides move) :
    ((perMove sides move).map fun outcome => if outcome.isDouble = flag then outcome.count else 0).sum =
      if flag then double move else count sides move - double move := by
  by_cases accepted : double move < count sides move <;> by_cases even : move % 2 = 0 <;> cases flag <;>
    simp only [double, even, ite_true, ite_false] at enough accepted <;>
    simp [perMove, ordinary, doubled, accepted, even, double] <;> omega

theorem ordinary_weight (sides move : Nat) (lower : 2 ≤ move) (upper : move ≤ 2 * sides) :
    ((perMove sides move).map fun outcome => if outcome.isDouble = false then outcome.count else 0).sum =
      (((DiceSums.pairs sides move).filter fun pair => pair.1 ≠ pair.2).card : Int) := by
  rw [per_move_weight _ _ false (count_ge_double sides move lower upper), DiceSums.non_doubles_card sides move lower upper]
  simp only [Bool.false_eq_true, ite_false, count_nat sides move lower upper, double_nat]
  have enough := count_ge_double sides move lower upper
  rw [count_nat sides move lower upper, double_nat] at enough
  omega

theorem double_weight (sides move : Nat) (lower : 2 ≤ move) (upper : move ≤ 2 * sides) :
    ((perMove sides move).map fun outcome => if outcome.isDouble = true then outcome.count else 0).sum =
      (((DiceSums.pairs sides move).filter fun pair => pair.1 = pair.2).card : Int) := by
  rw [per_move_weight _ _ true (count_ge_double sides move lower upper), DiceSums.doubles_card sides move lower upper]
  simp [double_nat]

theorem per_move_member (sides move : Int) (outcome : Outcome) (member : outcome ∈ perMove sides move) :
    outcome.move = move ∧ 0 < outcome.count := by
  by_cases accepted : double move < count sides move <;> by_cases even : move % 2 = 0 <;>
    simp only [perMove, ordinary, doubled, accepted, even, ite_true, ite_false, List.mem_append,
      List.mem_cons, List.not_mem_nil, or_false, false_or] at member
  all_goals first
    | contradiction
    | (rcases member with rfl | rfl <;> dsimp <;> omega)

theorem all_member (sides : Nat) (positive : 0 < sides) (outcome : Outcome) (member : outcome ∈ all sides) :
    2 ≤ outcome.move ∧ outcome.move ≤ 2 * sides ∧ 0 < outcome.count := by
  obtain ⟨move, visited, emitted⟩ := List.mem_flatMap.mp member
  have bounds := List.mem_range'.mp visited
  have fields := per_move_member sides move outcome emitted
  rw [fields.1]
  exact ⟨by omega, by omega, fields.2⟩

end solvers.«84».outcomes
