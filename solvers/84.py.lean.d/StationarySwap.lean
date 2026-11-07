import solvers.«84.py.lean.d».StationaryHeapView

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
  have afterView := view.reindex (Equiv.swap first second)
  refine ⟨?_, afterView⟩
  let saved := setLocal env "pivot_values" (.ref (addresses second))
  let intermediate := heap.set root
    (.list ((List.ofFn fun row => Mutable.Value.ref (addresses row)).set second.val (.ref (addresses first))))
  have savedMatrix : saved "matrix" = some (.ref root) := by simpa [saved, setLocal] using getMatrix
  have savedFirst : saved "column" = some (.scalar (.int first.val)) := by simpa [saved, setLocal] using getFirst
  have savedSecond : saved "pivot_row" = some (.scalar (.int second.val)) := by simpa [saved, setLocal] using getSecond
  have values := BlockContinues.assign (name := "pivot_values")
    (view.row_result program env globals (.name "matrix") (.name "pivot_row") second
      (.lookup getMatrix) (.lookup getSecond) (by intros; simp))
  have firstWrite : BlockContinues program saved
      [.assign [.subscript (.name "matrix") (.name "pivot_row")] (.subscript (.name "matrix") (.name "column"))]
      heap saved intermediate globals := by
    apply BlockContinues.setItem (by intros; simp)
      (view.row_result program saved globals (.name "matrix") (.name "column") first
        (.lookup savedMatrix) (.lookup savedFirst) (by intros; simp))
      (.cons (.lookup savedMatrix) (.cons (.lookup savedSecond) .nil))
    exact (view.swap_operations first second).1
  have secondWrite : BlockContinues program saved
      [.assign [.subscript (.name "matrix") (.name "column")] (.name "pivot_values")]
      intermediate saved (swappedHeap heap root addresses first second) globals := by
    apply BlockContinues.setItem (by intros; simp) (.lookup rfl)
      (.cons (.lookup savedMatrix) (.cons (.lookup savedFirst) .nil))
    exact (view.swap_operations first second).2
  have pivot : Evaluates program saved (.subscript (.name "pivot_values") (.name "column"))
      (swappedHeap heap root addresses first second) (.scalar (.int (matrix second first.castSucc)))
      (swappedHeap heap root addresses first second) globals := by
    simpa only [Function.comp_apply, Equiv.swap_apply_left] using
      afterView.entry_result program saved globals (.name "pivot_values") (.name "column") first first.castSucc
        (.lookup (by simp [saved, setLocal])) (.lookup savedFirst) (by intros; simp)
  exact values.append (firstWrite.append (secondWrite.append (BlockContinues.assign pivot)))

end solvers.«84».stationary_weights
