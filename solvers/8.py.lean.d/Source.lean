import «.formalsw».solvers.«8.py.ast»
import lang.py.FormalSW.HeapCalls

/-! Source fragments and elementary scan facts shared by correctness and cost
proofs. Kept separate from the mathematical window-product development so cost
checks do not need to load that development. -/
namespace solvers.«8»
open FormalSW.PyAst

def ints (xs : List Int) : List Mutable.Value := xs.map fun x => .scalar (.int x)

def scanTest (name : String) (zeros : Bool) : Expr :=
  .boolOp .and [
    .compare (.name name) [(.lt, .name "digit_count")],
    .compare (.subscript (.name "digits") (.name name))
      [(if zeros then .eq else .notEq, .constant (.int 0))]]

def scanLoop (name : String) (zeros : Bool) : Stmt :=
  .whileDo (scanTest name zeros) [.augAssign (.name name) .add (.constant (.int 1))] []

theorem setLocal_shadow (env : HeapCalls.Env) (name : String) (a b : Mutable.Value) :
    HeapCalls.setLocal (HeapCalls.setLocal env name a) name b = HeapCalls.setLocal env name b := by
  funext query
  by_cases equal : query = name <;> simp [HeapCalls.setLocal, equal]

theorem setLocal_current (env : HeapCalls.Env) (name : String) (value : Mutable.Value)
    (current : env name = some value) : HeapCalls.setLocal env name value = env := by
  funext query
  by_cases equal : query = name <;> simp_all [HeapCalls.setLocal]

theorem scan_bounds (digits : List Int) (start : Nat) (zeros : Bool) (bounded : start ≤ digits.length) :
    ∃ stop, start ≤ stop ∧ ∃ upper : stop ≤ digits.length,
      (∀ index, start ≤ index → (bound : index < stop) →
        (decide (digits[index]'(by omega) = 0) == zeros) = true) ∧
      (∀ bound : stop < digits.length, (decide (digits[stop]'bound = 0) == zeros) = false) := by
  by_cases valid : start < digits.length
  · by_cases selected : (decide (digits[start] = 0) == zeros) = true
    · obtain ⟨stop, ordered, bounded, inside, outside⟩ := scan_bounds digits (start + 1) zeros (by omega)
      refine ⟨stop, by omega, bounded, ?_, outside⟩
      intro index lower upper
      by_cases equal : index = start
      · subst index; exact selected
      · exact inside index (by omega) upper
    · exact ⟨start, Nat.le_refl _, bounded, by intros; omega, by intros; simpa using selected⟩
  · exact ⟨start, Nat.le_refl _, bounded, by intros; omega, by intros; omega⟩
termination_by digits.length - start

def productBody : List Stmt :=
  [.augAssign (.name "prod") .mult (.subscript (.name "digits") (.name "t"))]

theorem divide (a b : Int) (heap : Mutable.Heap) (nonzero : b ≠ 0) :
    Mutable.binary .floorDiv (.scalar (.int a)) (.scalar (.int b)) heap =
      some (.ok (.scalar (.int (a.fdiv b))), heap) := by
  by_cases positive : 0 < b
  · simp [Mutable.binary, scalarBinary, nonzero, positive,
      Int.fdiv_eq_ediv_of_nonneg a (by omega : 0 ≤ b)]
  · have quotient : a.fdiv b = (-a) / (-b) := by
      rw [← Int.neg_fdiv_neg, Int.fdiv_eq_ediv_of_nonneg _ (by omega : 0 ≤ -b)]
    simp [Mutable.binary, scalarBinary, nonzero, positive, quotient]

def rollExpression : Expr :=
  .binOp (.binOp (.name "prod") .floorDiv
    (.subscript (.name "digits") (.binOp (.name "t") .sub (.name "k")))) .mult
    (.subscript (.name "digits") (.name "t"))

def updateBest : Stmt :=
  .ifThen (.compare (.name "prod") [(.gt, .name "best")]) [.assign [.name "best"] (.name "prod")] []

def rollingBody : List Stmt := [.assign [.name "prod"] rollExpression, updateBest]

def segmentBody : List Stmt := [
  .assign [.name "prod"] (.constant (.int 1)),
  .forIn (.name "t") (.call (.name "range") [.name "i", .binOp (.name "i") .add (.name "k")]) productBody [],
  updateBest,
  .forIn (.name "t") (.call (.name "range") [.binOp (.name "i") .add (.name "k"), .name "j"]) rollingBody []]

def processSegment : List Stmt := [
  .assign [.name "seg_len"] (.binOp (.name "j") .sub (.name "i")),
  .ifThen (.compare (.name "seg_len") [(.gtE, .name "k")]) segmentBody []]

def outerBody : List Stmt := [
  scanLoop "i" true,
  .ifThen (.compare (.name "i") [(.gtE, .name "digit_count")]) [.breakLoop] [],
  .assign [.name "j"] (.name "i"),
  scanLoop "j" false] ++ processSegment ++ [.assign [.name "i"] (.name "j")]

def outerLoop : Stmt :=
  .whileDo (.compare (.name "i") [(.lt, .name "digit_count")]) outerBody []


end solvers.«8»
