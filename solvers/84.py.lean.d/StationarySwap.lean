import solvers.«84.py.lean.d».StationaryHeapView
import ProjectEulerSolutions.Lib.MatrixRowSwapCalls

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def eliminationBody : List Stmt :=
  match (executableBody definition.2)[(4 : Nat)]? with
  | some (.forIn _ _ body _) => body
  | _ => []

def swapBody : List Stmt :=
  [.assign [.name "pivot_values"] (.subscript (.name "matrix") (.name "pivot_row")),
   .assign [.subscript (.name "matrix") (.name "pivot_row")] (.subscript (.name "matrix") (.name "column")),
   .assign [.subscript (.name "matrix") (.name "column")] (.name "pivot_values"),
   .assign [.name "pivot"] (.subscript (.name "pivot_values") (.name "column"))]

theorem swap_source : (eliminationBody.drop 2).take 4 = swapBody := rfl

def swappedHeap (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat) (first second : Fin 120) : Mutable.Heap :=
  heap.set root (.list (List.ofFn fun row => .ref (addresses (Equiv.swap first second row))))

def swappedEnv (env : HeapCalls.Env) (addresses : Fin 120 → Nat)
    (matrix : Matrix (Fin 120) (Fin 121) Int) (first second : Fin 120) : HeapCalls.Env :=
  setLocal (setLocal env "pivot_values" (.ref (addresses second))) "pivot" (.scalar (.int (matrix second first.castSucc)))

/-- Execute both pointer assignments before reading the chosen pivot through its saved row reference. -/
theorem swap_result (matrix : Matrix (Fin 120) (Fin 121) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin 120 → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses)
    (first second : Fin 120) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getFirst : env "column" = some (.scalar (.int first.val)))
    (getSecond : env "pivot_row" = some (.scalar (.int second.val))) :
    BlockContinues program env swapBody heap (swappedEnv env addresses matrix first second)
      (swappedHeap heap root addresses first second) globals ∧
    MutableIntegerMatrix.View (IntegerRowSwap.swap matrix first second)
      (swappedHeap heap root addresses first second) root (addresses ∘ Equiv.swap first second) := by
  obtain ⟨swapped, afterView⟩ := MatrixRowSwapCalls.result program matrix heap root addresses view first second env globals
    getMatrix getFirst getSecond
  refine ⟨?_, afterView⟩
  let saved := MatrixRowSwapCalls.savedEnv env addresses second
  have savedFirst : saved "column" = some (.scalar (.int first.val)) := by
    simpa [saved, MatrixRowSwapCalls.savedEnv, setLocal] using getFirst
  have pivot : Evaluates program saved (.subscript (.name "pivot_values") (.name "column"))
      (swappedHeap heap root addresses first second) (.scalar (.int (matrix second first.castSucc)))
      (swappedHeap heap root addresses first second) globals := by
    simpa only [IntegerRowSwap.swap, Function.comp_apply, Equiv.swap_apply_left] using
      afterView.entry_result program saved globals (.name "pivot_values") (.name "column") first first.castSucc
        (.lookup (by simp [saved, MatrixRowSwapCalls.savedEnv, setLocal])) (.lookup savedFirst) (by intros; simp)
  exact swapped.append (BlockContinues.assign pivot)

end solvers.«84».stationary_weights
