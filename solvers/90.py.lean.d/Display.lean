import solvers.«90.py.lean.d».Rotation
import lang.py.FormalSW.BooleanCalls
import lang.py.FormalSW.UnpackReturnCalls

/-! The actual display predicate reads every required tuple, stopping at the first failure. -/
namespace solvers.«90».can_display
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.DigitDice
set_option Elab.async false

structure Pair where
  x : Nat
  y : Nat
  address : Nat

def Pair.model (pair : Pair) : Nat × Nat := (pair.x, pair.y)
def Pair.reference (pair : Pair) : Mutable.Value := .ref pair.address
def Pair.Valid (pair : Pair) (heap : Mutable.Heap) : Prop :=
  heap[pair.address]? = some (.tuple [number pair.x, number pair.y])

def accepted (a b : Nat) (pair : Pair) : Bool :=
  (a &&& 2 ^ pair.x != 0) && (b &&& 2 ^ pair.y != 0) ||
    (a &&& 2 ^ pair.y != 0) && (b &&& 2 ^ pair.x != 0)

def code : List Stmt :=
  match findFunctionIn program.body "can_display" with
  | some (_, body) => executableBody body
  | none => []

def loopCode : List Stmt :=
  match (code[0]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private theorem bit_and_truth (a b : Nat) (left right : String) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getLeft : env left = some (number a)) (getRight : env right = some (number b)) :
    EvaluatesTruth program env (.binOp (.name left) .bitAnd (.name right))
      heap (a &&& b != 0) globals := by
  refine ⟨number (a &&& b), ?_, ?_⟩
  · exact Evaluates.binary (Evaluates.lookup getLeft) (Evaluates.lookup getRight) rfl
  · simp [Mutable.truth, scalarTruth, bne, beq_eq_decide]

private theorem power (digit : Nat) (name : String) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (found : env name = some (number digit)) :
    Evaluates program env (.binOp (.constant (.int 1)) .leftShift (.name name))
      heap (number (2 ^ digit)) heap globals := by
  apply Evaluates.binary Evaluates.constant (Evaluates.lookup found)
  simp [Mutable.binary, scalarShiftLeft, integerShiftLeft, number,
    show ¬ (digit : Int) < 0 by omega]

private theorem condition (a b : Nat) (pair : Pair) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getA : env "mask_a" = some (number a)) (getB : env "mask_b" = some (number b))
    (getX : env "bx" = some (number (2 ^ pair.x)))
    (getY : env "by" = some (number (2 ^ pair.y))) :
    Evaluates program env
      (.unaryOp .not (.boolOp .or [
        .boolOp .and [.binOp (.name "mask_a") .bitAnd (.name "bx"),
          .binOp (.name "mask_b") .bitAnd (.name "by")],
        .boolOp .and [.binOp (.name "mask_a") .bitAnd (.name "by"),
          .binOp (.name "mask_b") .bitAnd (.name "bx")]]))
      heap (.scalar (.bool (!accepted a b pair))) heap globals := by
  apply EvaluatesTruth.not
  exact ((bit_and_truth a _ _ _ env heap globals getA getX).and
    (bit_and_truth b _ _ _ env heap globals getB getY)).or
      ((bit_and_truth a _ _ _ env heap globals getA getY).and
        (bit_and_truth b _ _ _ env heap globals getB getX))

/-- Each iteration unpacks the caller's tuple and preserves the complete input heap. -/
theorem loop (a b address : Nat) (pairs remaining : List Pair) (position : Nat)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (pairs.map Pair.reference)))
    (valid : ∀ pair ∈ remaining, pair.Valid heap)
    (suffix : pairs.drop position = remaining)
    (getA : env "mask_a" = some (number a)) (getB : env "mask_b" = some (number b)) :
    BlockReturns program env
      [.unpackStepLoop ["x", "y"] (.ref address) position loopCode [], code[1]'(by decide)]
      heap (.scalar (.bool (remaining.all (accepted a b)))) heap globals := by
  induction remaining generalizing position env with
  | nil =>
      apply BlockReturns.unpack_done (items := pairs.map Pair.reference)
      · simp [sequenceValue, Mutable.sequence, found]
      · have missing : pairs[position]? = none :=
          List.getElem?_eq_none_iff.mpr (List.drop_eq_nil_iff.mp suffix)
        simp [missing]
      · exact BlockReturns.return_value Evaluates.constant
  | cons pair rest ih =>
      have current := valid pair (by simp)
      change heap[pair.address]? = some (.tuple [number pair.x, number pair.y]) at current
      have item : pairs[position]? = some pair := by
        simpa [suffix] using (List.head?_drop (l := pairs) (i := position)).symm
      have tail : pairs.drop (position + 1) = rest := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      let bound := assignNames env ["x", "y"] [number pair.x, number pair.y]
      let withX := setLocal bound "bx" (number (2 ^ pair.x))
      let withY := setLocal withX "by" (number (2 ^ pair.y))
      have boundX : bound "x" = some (number pair.x) := by simp [bound, assignNames, setLocal]
      have boundY : withX "y" = some (number pair.y) := by simp [withX, bound, assignNames, setLocal]
      have nextA : withY "mask_a" = some (number a) := by
        simpa [withY, withX, bound, assignNames, setLocal] using getA
      have nextB : withY "mask_b" = some (number b) := by
        simpa [withY, withX, bound, assignNames, setLocal] using getB
      have tested := condition a b pair withY heap globals nextA nextB
        (by simp [withY, withX, setLocal]) (by simp [withY, setLocal])
      apply BlockReturns.unpack_step (items := pairs.map Pair.reference)
        (value := pair.reference) (values := [number pair.x, number pair.y])
      · simp [sequenceValue, Mutable.sequence, found]
      · simp [item]
      · simp [Pair.reference, Mutable.sequence, current]
      · rfl
      · apply BlockReturns.assign (power pair.x "x" bound heap globals boundX)
        apply BlockReturns.assign (power pair.y "y" withX heap globals boundY)
        cases ok : accepted a b pair with
        | false =>
            simp only [List.all_cons, ok, Bool.false_and]
            apply BlockReturns.branch (flag := true) (by simpa only [ok] using tested) rfl
            exact BlockReturns.return_value Evaluates.constant
        | true =>
            simp only [List.all_cons, ok, Bool.true_and]
            apply BlockReturns.branch (flag := false) (by simpa only [ok] using tested) rfl
            exact ih (position + 1) withY (fun p mem => valid p (by simp [mem])) tail nextA nextB

/-- The Python predicate equals the bitmask model for every finite list of digit pairs. -/
theorem result (a b address : Nat) (pairs : List Pair) (heap : Mutable.Heap)
    (found : heap[address]? = some (.list (pairs.map Pair.reference)))
    (valid : ∀ pair ∈ pairs, pair.Valid heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "can_display" [number a, number b, .ref address] heap
      (.scalar (.bool (maskDisplays a b (pairs.map Pair.model)))) heap globals := by
  have model : maskDisplays a b (pairs.map Pair.model) = pairs.all (accepted a b) := by
    simp only [maskDisplays, List.all_map]
    rfl
  rw [model]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.for_unpack (by intros; simp) rfl
    (Evaluates.lookup (by cases globals <;> rfl))
    (items := pairs.map Pair.reference)
  · simp [Mutable.sequence, found]
  · exact loop a b address pairs pairs 0 heap globals _ found valid rfl
      (by cases globals <;> rfl) (by cases globals <;> rfl)

end solvers.«90».can_display
