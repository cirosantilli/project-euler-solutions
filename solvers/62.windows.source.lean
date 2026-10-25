import ProjectEulerSolutions.Lib.CubicWindows
import solvers.«62.search.source»
import solvers.«62.reference.source»

/-! A successful completed reference window executes both actual Python sources. -/
set_option Elab.async false
namespace solvers.«62».closed_window
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.CubicGroupCounts ProjectEulerSolutions.CubicSearch ProjectEulerSolutions.CubicWindows
open data.«project-euler-statements».data.py.project_euler_statements.p62
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

theorem cubes_prior (limit : Nat) : reference_model.cubes limit = prior (limit + 1) := by
  simp [reference_model.cubes, prior, cube, pow_succ]

theorem reference_choice (limit : Nat) (target : Int) :
    choose target (collect (prior (limit + 1))) = naive.answer limit target := by
  rw [← cubes_prior, reference_model.answer_reference]

theorem witness (limit : Nat) (target : Int) (closed : width limit < width (limit + 1))
    (success : naive.answer limit target ≠ 0) : Returns target start (naive.answer limit target) := by
  have successful : choose target (collect (prior (limit + 1))) ≠ 0 := by
    simpa only [reference_choice] using success
  have returned := returns_closed target (limit + 1) (by omega)
    (by simpa only [Nat.add_sub_cancel] using closed) successful
  simpa only [reference_choice] using returned

/-- The heaps may allocate different objects, but both sources return the same scalar. -/
theorem result (limit : Nat) (target : Int) (caller : Mutable.Heap)
    (closed : width limit < width (limit + 1)) (success : naive.answer limit target ≠ 0) :
    ∃ finalHeap extra,
      ReturnsValue Solvers.P62Ast.module "solve" [number target] caller
        (number (naive.answer limit target)) finalHeap ∧
      finalHeap.take caller.length = caller ∧
      ReturnsValue P62Ast.module "naive" [number limit, number target] caller
        (number (naive.answer limit target)) (caller ++ extra) := by
  obtain ⟨finalHeap, optimized, preserved⟩ := search.result target _ caller (witness limit target closed success)
  obtain ⟨extra, reference⟩ := naive.result limit target caller
  exact ⟨finalHeap, extra, optimized, preserved, reference⟩

end solvers.«62».closed_window

namespace solvers.«62»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open data.«project-euler-statements».data.py.project_euler_statements.p62

theorem closed_window_source_equivalence (limit : Nat) (target : Int) (caller : Mutable.Heap)
    (closed : ProjectEulerSolutions.CubicSearch.width limit < ProjectEulerSolutions.CubicSearch.width (limit + 1))
    (success : naive.answer limit target ≠ 0) :
    ∃ finalHeap extra,
      ReturnsValue Solvers.P62Ast.module "solve" [.scalar (.int target)] caller
        (.scalar (.int (naive.answer limit target))) finalHeap ∧
      finalHeap.take caller.length = caller ∧
      ReturnsValue P62Ast.module "naive" [.scalar (.int limit), .scalar (.int target)] caller
        (.scalar (.int (naive.answer limit target))) (caller ++ extra) :=
  closed_window.result limit target caller closed success

end solvers.«62»
