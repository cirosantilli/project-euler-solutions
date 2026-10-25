import ProjectEulerSolutions.Lib.CubicGroupCounts
import data.«project-euler-statements».data.py.project_euler_statements.«p62.py»

/-! The ordered group summary agrees with the independent exhaustive reference. -/
set_option Elab.async false
namespace solvers.«62».reference_model
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.CubicPermutations ProjectEulerSolutions.CubicGroups ProjectEulerSolutions.CubicGroupCounts
open data.«project-euler-statements».data.py.project_euler_statements.p62
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

def cubes (limit : Int) : List Nat := (List.range' 1 limit.toNat).map (fun n : Nat => n ^ 3)
def accepted (limit target : Int) (n : Nat) : Bool := decide ((cube_perm_count.answer limit n : Int) = target)

theorem occurrences_reference (limit : Int) (n : Nat) (positive : 0 < n) :
    occurrences (signature (n ^ 3)) (cubes limit) = cube_perm_count.answer limit n := by
  unfold occurrences cubes cube_perm_count.answer cube_perm_count.selected
  rw [List.filter_map, List.length_map]
  congr 1
  apply List.filter_congr
  intro i member
  have positiveI : 0 < i := by
    have bound := (List.mem_range'_1.mp member).1
    omega
  unfold cube_perm_count.accepts
  apply Bool.eq_iff_iff.mpr
  simp only [Function.comp_apply, decide_eq_true_eq]
  exact signature_eq_sorted (i ^ 3) (n ^ 3) (Nat.pow_pos positiveI) (Nat.pow_pos positive)

theorem eligible_reference (limit target : Int) :
    eligible target (cubes limit) =
      ((List.range' 1 limit.toNat).filter (accepted limit target)).map (fun n : Nat => ((n ^ 3 : Nat) : Int)) := by
  unfold eligible cubes
  rw [List.filterMap_map]
  conv_rhs => rw [← List.filterMap_eq_map', List.filterMap_filter]
  apply List.filterMap_congr
  intro n member
  have positive : 0 < n := by
    have bound := (List.mem_range'_1.mp member).1
    omega
  simp only [Function.comp_apply]
  have counted : occurrences (signature (n ^ 3)) ((List.range' 1 limit.toNat).map (fun n : Nat => n ^ 3)) =
      cube_perm_count.answer limit n := by simpa only [cubes] using occurrences_reference limit n positive
  rw [counted]
  simp [accepted]

theorem search_reference (limit target : Int) (first count : Nat) :
    naive.search limit target first count = firstCube (accepted limit target) first count := by
  induction count generalizing first with
  | zero => rfl
  | succ count ih =>
      by_cases same : (cube_perm_count.answer limit first : Int) = target
      · simp [naive.search, firstCube, accepted, same]
      · simp [naive.search, firstCube, accepted, same, ih]

/-- First-match traversal and minimum selection agree for every signed bound and target. -/
theorem answer_reference (limit target : Int) :
    naive.answer limit target = choose target (collect (cubes limit)) := by
  rw [naive.answer, search_reference, firstCube_min, choose_collect, eligible_reference]

theorem summary_reference (limit target : Int) (groups : List Group) (summary : Summary (cubes limit) groups) :
    choose target groups = naive.answer limit target := by
  rw [answer_reference, choose_collect, choose, summary.minimum_eq_eligible]

/-- Any complete mathematical summary computes the actual independent reference source result. -/
theorem source_result (limit target : Int) (caller : Mutable.Heap) (groups : List Group)
    (summary : Summary (cubes limit) groups) :
    ∃ extra, ReturnsValue P62Ast.module "naive" [number limit, number target] caller
      (number (choose target groups)) (caller ++ extra) := by
  obtain ⟨extra, returned⟩ := naive.result limit target caller
  exact ⟨extra, by simpa only [summary_reference limit target groups summary] using returned⟩

end solvers.«62».reference_model

namespace solvers.«62»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

theorem grouped_reference_source_result (limit target : Int) (caller : Mutable.Heap)
    (groups : List ProjectEulerSolutions.CubicGroups.Group)
    (summary : ProjectEulerSolutions.CubicGroupCounts.Summary (reference_model.cubes limit) groups) :
    ∃ extra, ReturnsValue P62Ast.module "naive" [.scalar (.int limit), .scalar (.int target)] caller
      (.scalar (.int (ProjectEulerSolutions.CubicGroupCounts.choose target groups))) (caller ++ extra) :=
  reference_model.source_result limit target caller groups summary

end solvers.«62»
