import solvers.«84.py.lean.d».StationaryTriangular
import ProjectEulerSolutions.Lib.MutableIntegerMatrixAccess

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

structure BackView (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat) (weights : Nat) : Prop where
  matrixView : MutableIntegerMatrix.View matrix heap root addresses
  weightsFound : heap[weights]? = some (.list (List.ofFn fun index => .scalar (.int (values index))))
  notRoot : weights ≠ root
  notRows : ∀ row, weights ≠ addresses row

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

def weightsObjects : Mutable.Heap :=
  [.list [.scalar (.int 0)], .list (List.ofFn fun _ : Fin 120 => .scalar (.int 0))]

theorem weights_value (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSize : env "size" = some (.scalar (.int 120))) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult (.name "size"))
      heap (.ref (heap.length + 1)) (heap ++ weightsObjects) globals := by
  apply Evaluates.binary (Evaluates.list (.cons Evaluates.constant .nil)) (.lookup getSize)
  simp only [weightsObjects, List.ofFn_const]
  simp [Mutable.binary, Mutable.array, Mutable.allocate, Except.map, List.append_assoc,
    show (BinOp.mult == BinOp.mult) = true from rfl]

theorem weights_view (matrix : Matrix (Fin 120) (Fin 121) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin 120 → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses) :
    BackView matrix (fun _ => 0) (heap ++ weightsObjects) root addresses (heap.length + 1) := by
  refine ⟨view.append _, ?_, ?_, ?_⟩
  · rw [List.getElem?_append_right (show heap.length ≤ heap.length + 1 by omega)]
    simp only [Nat.add_sub_cancel_left, weightsObjects]
    rfl
  · have inside := (List.getElem?_eq_some_iff.mp view.found).1
    omega
  · intro row
    have inside := (List.getElem?_eq_some_iff.mp (view.rows row)).1
    omega

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
