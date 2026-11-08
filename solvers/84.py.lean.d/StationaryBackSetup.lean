import solvers.«84.py.lean.d».StationaryTriangular
import ProjectEulerSolutions.Lib.MatrixBackBufferCalls

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

export ProjectEulerSolutions.MatrixBackBufferCalls (BackView weightsObjects weights_view)
abbrev weights_value := ProjectEulerSolutions.MatrixBackBufferCalls.weights_value program

theorem scale_value (matrix : Matrix (Fin 120) (Fin 121) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin 120 → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getMatrix : env "matrix" = some (.ref root)) :
    Evaluates program env (.subscript (.subscript (.name "matrix") (.unaryOp .uSub (.constant (.int 1))))
      (.unaryOp .uSub (.constant (.int 2)))) heap (.scalar (.int (matrix 119 119))) heap globals := by
  have row : Evaluates program env (.subscript (.name "matrix") (.unaryOp .uSub (.constant (.int 1))))
      heap (.ref (addresses 119)) heap globals := by
    apply Evaluates.subscript (by intros; simp) (.lookup getMatrix) (Evaluates.unary Evaluates.constant rfl)
    exact view.row_fromEnd 119
  apply Evaluates.subscript (by intros; simp) row (Evaluates.unary Evaluates.constant rfl)
  exact view.entry_fromEnd 119 119

def backSetupEnv (env : HeapCalls.Env) (heap : Mutable.Heap) (scale : Int) : HeapCalls.Env :=
  setLocal (setLocal env "scale" (.scalar (.int scale))) "weights" (.ref (heap.length + 1))

/-- Read the proved nonzero final diagonal and allocate the actual zero-filled reverse-substitution buffer. -/
theorem back_setup_result {sides : Nat} (state : Bareiss.State (Monopoly.augmentedMatrix sides) 119)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat)
    (view : MutableIntegerMatrix.View state.matrix heap root addresses)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root)) (getSize : env "size" = some (.scalar (.int 120))) :
    BlockContinues program env ((executableBody definition.2).drop 5 |>.take 3) heap
      (backSetupEnv env heap (state.matrix 119 119)) (heap ++ weightsObjects) globals ∧
    BackView state.matrix (fun _ => 0) (heap ++ weightsObjects) root addresses (heap.length + 1) := by
  refine ⟨?_, weights_view state.matrix heap root addresses view⟩
  have scale := BlockContinues.assign (name := "scale") (scale_value state.matrix heap root addresses view env globals getMatrix)
  let scaled := setLocal env "scale" (.scalar (.int (state.matrix 119 119)))
  have nonzero : state.matrix 119 119 ≠ 0 := state.last_nonzero
  have guard : BlockContinues program scaled (((executableBody definition.2).drop 6).take 1) heap scaled heap globals := by
    apply BlockContinues.branch (flag := false) (value := .scalar (.bool false))
      (Evaluates.compare (.lookup rfl) Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, nonzero]))
      rfl BlockContinues.nil
  have allocated := BlockContinues.assign (name := "weights")
    (weights_value heap scaled globals (by simpa [scaled, setLocal] using getSize))
  exact scale.append (guard.append allocated)

end solvers.«84».stationary_weights
