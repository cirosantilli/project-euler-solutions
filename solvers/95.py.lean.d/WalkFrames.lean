import solvers.«95.py.lean.d».Step
import solvers.«95.py.lean.d».Cycle
import ProjectEulerSolutions.Lib.PathPositions

/-! The walk's table and local bindings survive cycle allocation and processed writes. -/
namespace solvers.«95».walk
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

variable {env finalEnv : HeapCalls.Env} {a : Addresses} {limit current : Nat} {best : Nat × Nat}

theorem Captured.frame (captured : Captured env a limit current best)
    (stable : ∀ name ∈ ["limit", "s", "processed", "path", "idx", "n", "best_len", "best_min"],
      finalEnv name = env name) : Captured finalEnv a limit current best where
  getLimit := (stable "limit" (by simp)).trans captured.getLimit
  getTable := (stable "s" (by simp)).trans captured.getTable
  getProcessed := (stable "processed" (by simp)).trans captured.getProcessed
  getPath := (stable "path" (by simp)).trans captured.getPath
  getPositions := (stable "idx" (by simp)).trans captured.getPositions
  getCurrent := (stable "n" (by simp)).trans captured.getCurrent
  getLength := (stable "best_len" (by simp)).trans captured.getLength
  getMinimum := (stable "best_min" (by simp)).trans captured.getMinimum

theorem Captured.ready (captured : Captured env a limit current best) (address : Nat) (values : List Nat) :
    Captured (cycle.ready env address values) a limit current best := by
  apply captured.frame
  intro name member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact cycle.ready_frame _ _ _ _ (by decide) (by decide) (by decide)

theorem Captured.winner (captured : Captured env a limit current best) (chosen : Nat × Nat)
    (length : finalEnv "best_len" = some (number chosen.1))
    (minimum : finalEnv "best_min" = some (number chosen.2))
    (stable : ∀ name, name ≠ "best_len" → name ≠ "best_min" → finalEnv name = env name) :
    Captured finalEnv a limit current chosen where
  getLimit := (stable "limit" (by decide) (by decide)).trans captured.getLimit
  getTable := (stable "s" (by decide) (by decide)).trans captured.getTable
  getProcessed := (stable "processed" (by decide) (by decide)).trans captured.getProcessed
  getPath := (stable "path" (by decide) (by decide)).trans captured.getPath
  getPositions := (stable "idx" (by decide) (by decide)).trans captured.getPositions
  getCurrent := (stable "n" (by decide) (by decide)).trans captured.getCurrent
  getLength := length
  getMinimum := minimum

variable {heap : Mutable.Heap} {f : Nat → Nat} {known : Finset Nat} {path : List Nat}
  {positions : NaturalDictionary.Entries}

theorem Found.append (found : Found heap a limit f known path positions) (extra : Mutable.Heap) :
    Found (heap ++ extra) a limit f known path positions where
  table := by rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found.table).1]; exact found.table
  processed := by rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found.processed).1]; exact found.processed
  path := by rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found.path).1]; exact found.path
  positions := by
    change (heap ++ extra)[a.positions]? = _
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found.positions).1]
    exact found.positions

theorem Found.marked (found : Found heap a limit f known path positions) (separate : a.Separated) :
    Found (heap.set a.processed (.list (processed.items limit (known ∪ path.toFinset))))
      a limit f (known ∪ path.toFinset) path positions where
  table := by rw [List.getElem?_set_ne separate.tableProcessed.symm]; exact found.table
  processed := List.getElem?_set_self (List.getElem?_eq_some_iff.mp found.processed).1
  path := by rw [List.getElem?_set_ne separate.processedPath]; exact found.path
  positions := by
    change (heap.set a.processed _)[a.positions]? = _
    rw [List.getElem?_set_ne separate.processedPositions]
    exact found.positions

end solvers.«95».walk
