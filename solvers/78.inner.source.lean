import «.formalsw».solvers.«78.py.ast»
import lang.py.FormalSW.LoopCalls
import ProjectEulerStatements.Lib.PartitionPentagonalTerms

/-! Source expressions shared by the exact and modular pentagonal loops. -/
namespace solvers.«78»
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.PartitionPentagonalLoop
set_option Elab.async false
variable {globals : Mutable.Globals}

abbrev program := Solvers.P78Ast.module
abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

namespace inner

def offsetExpression (upper : Bool) : Expr :=
  .binOp (.binOp (.name "k") .mult
    (.binOp (.binOp (.constant (.int 3)) .mult (.name "k"))
      (if upper then .add else .sub) (.constant (.int 1)))) .floorDiv (.constant (.int 2))

def signTest : Expr := .compare (.binOp (.name "k") .mod (.constant (.int 2)))
  [(.eq, .constant (.int 1))]

def signExpression : Expr := .ifExp signTest (.constant (.int 1))
  (.unaryOp .uSub (.constant (.int 1)))

def addExpression (boundName offsetName : String) : Expr :=
  .binOp (.name "sign") .mult
    (.subscript (.name "p") (.binOp (.name boundName) .sub (.name offsetName)))

def body (boundName : String) : List Stmt :=
  [.assign [.name "g1"] (offsetExpression false),
   .ifThen (.compare (.name "g1") [(.gt, .name boundName)]) [.breakLoop] [],
   .assign [.name "sign"] signExpression,
   .augAssign (.name "total") .add (addExpression boundName "g1"),
   .assign [.name "g2"] (offsetExpression true),
   .ifThen (.compare (.name "g2") [(.ltE, .name boundName)])
     [.augAssign (.name "total") .add (addExpression boundName "g2")] [],
   .augAssign (.name "k") .add (.constant (.int 1))]

def statement (boundName : String) : Stmt := .whileDo (.constant (.bool true)) (body boundName) []

abbrev solveCode : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
abbrev solveBody : List Stmt := match (solveCode[2]? : Option Stmt) with
  | some (.whileDo _ instructions _) => instructions | _ => []
abbrev uptoCode : List Stmt := executableBody ((findFunctionIn program.body "partitions_upto").getD ([], [])).2
abbrev uptoBody : List Stmt := match (uptoCode[2]? : Option Stmt) with
  | some (.forIn _ _ instructions _) => instructions | _ => []

theorem solve_source : solveBody[2]? = some (statement "n") := rfl
theorem upto_source : uptoBody[2]? = some (statement "i") := rfl

def writes : List String := ["g1", "sign", "total", "g2", "k"]
def Keeps (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ∉ writes → after name = before name

theorem Keeps.refl (env : HeapCalls.Env) : Keeps env env := by intro _ _; rfl
theorem Keeps.trans (first : Keeps before middle) (second : Keeps middle after) : Keeps before after :=
  fun name untouched => (second name untouched).trans (first name untouched)

theorem Keeps.set (env : HeapCalls.Env) (name : String) (value : Mutable.Value) (written : name ∈ writes) :
    Keeps env (setLocal env name value) := by
  intro other untouched
  have different : other ≠ name := by intro same; subst other; contradiction
  simp [setLocal, different]

theorem offset_result (upper : Bool) (k : Nat) (heap : Heap) (env : HeapCalls.Env)
    (getK : env "k" = some (number k)) :
    Evaluates (globals := globals) program env (offsetExpression upper) heap
      (number (if upper then second k else first k)) heap := by
  apply Evaluates.binary (av := number ((k : Int)*(if upper then 3*(k : Int)+1 else 3*(k : Int)-1)))
    (bv := number 2) (middle := heap) (afterRight := heap) (finalHeap := heap)
  · apply Evaluates.binary (av := number k)
      (bv := number (if upper then 3*(k : Int)+1 else 3*(k : Int)-1))
      (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup getK)
    · cases upper <;> apply Evaluates.binary
        (Evaluates.binary Evaluates.constant (Evaluates.lookup getK) rfl) Evaluates.constant rfl
    · cases upper <;> rfl
  · exact Evaluates.constant
  · cases upper <;> simp [Mutable.binary, scalarBinary, number, first_cast, second_cast]

theorem sign_result (k : Nat) (heap : Heap) (env : HeapCalls.Env)
    (getK : env "k" = some (number k)) :
    Evaluates (globals := globals) program env signExpression heap (number (sign k)) heap := by
  have tested : Evaluates (globals := globals) program env signTest heap
      (flag (decide ((k : Int)%2 = 1))) heap := by
    apply Evaluates.compare (av := number ((k : Int)%2)) (bv := number 1)
      (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.binary (Evaluates.lookup getK) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases odd : (k : Int)%2 = 1
  · simp only [sign, if_pos odd]
    exact Evaluates.conditional (value := flag true) (flag := true)
      (by simpa only [odd, decide_true] using tested) rfl Evaluates.constant
  · simp only [sign, if_neg odd]
    apply Evaluates.conditional (value := flag false) (flag := false)
      (by simpa only [odd, decide_false] using tested) rfl
    exact Evaluates.unary Evaluates.constant rfl

/-- Each signed table read is within the already computed positive-index prefix. -/
theorem add_result (boundName offsetName : String) (n offset address : Nat)
    (values : Nat → Int) (items : List Mutable.Value) (heap : Heap) (env : HeapCalls.Env)
    (total coefficient : Int) (positive : 0 < offset) (inside : offset ≤ n)
    (found : heap[address]? = some (.list items))
    (entries : ∀ j < n, items[j]? = some (number (values j)))
    (getBound : env boundName = some (number n)) (getOffset : env offsetName = some (number offset))
    (getP : env "p" = some (.ref address)) (getSign : env "sign" = some (number coefficient))
    (getTotal : env "total" = some (number total)) :
    BlockContinues (globals := globals) program env
      [.augAssign (.name "total") .add (addExpression boundName offsetName)] heap
      (setLocal env "total" (number (total + coefficient * values (n-offset)))) heap := by
  have selected := entries (n-offset) (by omega)
  have read : Mutable.getItem (.ref address) (number (n-offset : Nat)) heap =
      some (.ok (number (values (n-offset))), heap) := by
    simpa only [(List.getElem?_eq_some_iff.mp selected).2] using
      Mutable.getItem_list heap address (n-offset) items found
        (List.getElem?_eq_some_iff.mp selected).1
  apply BlockContinues.augAssign (middle := heap) (finalHeap := heap)
    (right := number (coefficient * values (n-offset))) getTotal
  · apply Evaluates.binary (av := number coefficient) (bv := number (values (n-offset)))
      (middle := heap) (afterRight := heap) (finalHeap := heap) (Evaluates.lookup getSign)
    · apply Evaluates.subscript (position := number (n-offset : Nat))
        (middle := heap) (afterIndex := heap) (finalHeap := heap)
        (ordinary := by intros; simp) (Evaluates.lookup getP)
      · exact Evaluates.binary (Evaluates.lookup getBound) (Evaluates.lookup getOffset)
          (by simp [Mutable.binary, scalarBinary, show (n : Int)-offset = (n-offset : Nat) by omega])
      · exact read
    · rfl
  · rfl

end inner
end solvers.«78»
