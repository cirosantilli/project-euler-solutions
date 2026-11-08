import ProjectEulerSolutions.Lib.MutableIntegerMatrixAccess
import ProjectEulerSolutions.Lib.IntegerRowSwap

/-! The three pointer assignments that install a selected pivot row. -/
namespace ProjectEulerSolutions.MatrixRowSwapCalls
open FormalSW.PyAst HeapCalls
set_option Elab.async false
variable {n m : Nat}

def body : List Stmt :=
  [.assign [.name "pivot_values"] (.subscript (.name "matrix") (.name "pivot_row")),
   .assign [.subscript (.name "matrix") (.name "pivot_row")] (.subscript (.name "matrix") (.name "column")),
   .assign [.subscript (.name "matrix") (.name "column")] (.name "pivot_values")]

def swappedHeap (heap : Mutable.Heap) (root : Nat) (addresses : Fin n → Nat) (first second : Fin n) : Mutable.Heap :=
  heap.set root (.list (List.ofFn fun row => .ref (addresses (Equiv.swap first second row))))

def savedEnv (env : HeapCalls.Env) (addresses : Fin n → Nat) (second : Fin n) : HeapCalls.Env :=
  setLocal env "pivot_values" (.ref (addresses second))

/-- Both writes retain the chosen row's saved reference and preserve distinct row identities. -/
theorem result (module : Module) (matrix : Matrix (Fin n) (Fin m) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin n → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses)
    (first second : Fin n) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getFirst : env "column" = some (.scalar (.int first.val)))
    (getSecond : env "pivot_row" = some (.scalar (.int second.val))) :
    BlockContinues module env body heap (savedEnv env addresses second)
      (swappedHeap heap root addresses first second) globals ∧
    MutableIntegerMatrix.View (IntegerRowSwap.swap matrix first second)
      (swappedHeap heap root addresses first second) root (addresses ∘ Equiv.swap first second) := by
  refine ⟨?_, view.reindex (Equiv.swap first second)⟩
  let saved := savedEnv env addresses second
  let intermediate := heap.set root
    (.list ((List.ofFn fun row => Mutable.Value.ref (addresses row)).set second.val (.ref (addresses first))))
  have savedMatrix : saved "matrix" = some (.ref root) := by simpa [saved, savedEnv, setLocal] using getMatrix
  have savedFirst : saved "column" = some (.scalar (.int first.val)) := by simpa [saved, savedEnv, setLocal] using getFirst
  have savedSecond : saved "pivot_row" = some (.scalar (.int second.val)) := by simpa [saved, savedEnv, setLocal] using getSecond
  have values := BlockContinues.assign (name := "pivot_values")
    (view.row_result module env globals (.name "matrix") (.name "pivot_row") second
      (.lookup getMatrix) (.lookup getSecond) (by intros; simp))
  have firstWrite : BlockContinues module saved
      [.assign [.subscript (.name "matrix") (.name "pivot_row")] (.subscript (.name "matrix") (.name "column"))]
      heap saved intermediate globals := by
    apply BlockContinues.setItem (by intros; simp)
      (view.row_result module saved globals (.name "matrix") (.name "column") first
        (.lookup savedMatrix) (.lookup savedFirst) (by intros; simp))
      (.cons (.lookup savedMatrix) (.cons (.lookup savedSecond) .nil))
    exact (view.swap_operations first second).1
  have secondWrite : BlockContinues module saved
      [.assign [.subscript (.name "matrix") (.name "column")] (.name "pivot_values")]
      intermediate saved (swappedHeap heap root addresses first second) globals := by
    apply BlockContinues.setItem (by intros; simp) (.lookup rfl)
      (.cons (.lookup savedMatrix) (.cons (.lookup savedFirst) .nil))
    exact (view.swap_operations first second).2
  exact values.append (firstWrite.append secondWrite)

end ProjectEulerSolutions.MatrixRowSwapCalls
