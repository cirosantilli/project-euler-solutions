import «.formalsw».solvers.«93.py.ast»
import ProjectEulerSolutions.Lib.RationalPair
import lang.py.FormalSW.HeapCalls

namespace solvers.«93».rational
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
set_option Elab.async false

abbrev program := Solvers.P93Ast.module
abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
abbrev object (pair : Pair) : Mutable.Object := .tuple [number pair.1, number pair.2]

def tail : List Stmt :=
  [.assign [.name "divisor"] (.call (.name "gcd") [.name "numerator", .name "denominator"]),
   .return (.tuple [.binOp (.name "numerator") .floorDiv (.name "divisor"),
     .binOp (.name "denominator") .floorDiv (.name "divisor")])]

theorem tail_source :
    (executableBody ((findFunctionIn program.body "rational").getD ([], [])).2).drop 1 = tail := rfl

theorem gcd_result (a b : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getA : env "numerator" = some (number a)) (getB : env "denominator" = some (number b)) :
    Evaluates program env (.call (.name "gcd") [.name "numerator", .name "denominator"])
      heap (number (a.gcd b)) heap globals := by
  apply Evaluates.library_call (.cons (Evaluates.lookup getA) (.cons (Evaluates.lookup getB) .nil)) rfl
  have libraryFound : libraryFunction program "gcd" = some Math.gcdCall := rfl
  simp [HeapCalls.library, libraryFound, Math.gcdCall_pair, number, scalarResult, Int.gcd]

theorem divide (name : String) (a divisor : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (positive : 0 < divisor)
    (getA : env name = some (number a)) (getD : env "divisor" = some (number divisor)) :
    Evaluates program env (.binOp (.name name) .floorDiv (.name "divisor"))
      heap (number (a / divisor)) heap globals := by
  apply Evaluates.binary (Evaluates.lookup getA) (Evaluates.lookup getD)
  simp [Mutable.binary, scalarBinary, number, positive, positive.ne']

/-- The gcd and the two exact divisions allocate one normalized result tuple. -/
theorem tail_result (a b : Int) (positive : 0 < b) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (getA : env "numerator" = some (number a))
    (getB : env "denominator" = some (number b)) :
    BlockReturns program env tail heap (.ref heap.length) (heap ++ [object (reduced a b)]) globals := by
  have divisorPositive : (0 : Int) < a.gcd b := by
    exact_mod_cast Int.gcd_pos_of_ne_zero_right a positive.ne'
  apply BlockReturns.assign (gcd_result a b heap env globals getA getB)
  apply BlockReturns.return_value
  apply Evaluates.tuple (nonempty := by simp)
  apply Evaluates.Arguments.cons
  · exact divide "numerator" a (a.gcd b) heap _ globals divisorPositive
      (by simpa [setLocal] using getA) (by simp [setLocal])
  apply Evaluates.Arguments.cons
  · exact divide "denominator" b (a.gcd b) heap _ globals divisorPositive
      (by simpa [setLocal] using getB) (by simp [setLocal])
  exact .nil

/-- The complete source handles either denominator sign and preserves the caller heap. -/
theorem result (a b : Int) (nonzero : b ≠ 0) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    ReturnsValue program "rational" [number a, number b] heap (.ref heap.length)
      (heap ++ [object (ProjectEulerSolutions.RationalPair.normalize a b)]) globals := by
  let args := ((findFunctionIn program.body "rational").getD ([], [])).1
  let source := ((findFunctionIn program.body "rational").getD ([], [])).2
  let entry := functionEnv args source [number a, number b] globals
  have getA : entry "numerator" = some (number a) := by cases globals <;> rfl
  have getB : entry "denominator" = some (number b) := by cases globals <;> rfl
  apply BlockReturns.function rfl rfl
  by_cases negative : b < 0
  · rw [ProjectEulerSolutions.RationalPair.normalize, if_pos negative]
    apply BlockReturns.branch (flag := true)
      (Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap) (Evaluates.lookup getB)
        Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, negative])) rfl
    apply BlockReturns.assign (value := number (-a))
    · exact Evaluates.unary (finalHeap := heap) (Evaluates.lookup getA) rfl
    apply BlockReturns.assign (value := number (-b))
    · exact Evaluates.unary (finalHeap := heap) (Evaluates.lookup (by simpa [setLocal] using getB)) rfl
    exact tail_result (-a) (-b) (by omega) heap _ globals (by simp [setLocal]) (by simp [setLocal])
  · rw [ProjectEulerSolutions.RationalPair.normalize, if_neg negative]
    apply BlockReturns.branch (flag := false)
      (Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap) (Evaluates.lookup getB)
        Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, negative])) rfl
    exact tail_result a b (by omega) heap entry globals getA getB

def Verified (a b : Int) (heap : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  ReturnsValue program "rational" [number a, number b] heap (.ref heap.length)
    (heap ++ [object ((Rat.divInt a b).num, ((Rat.divInt a b).den : Int))]) globals

theorem verified (a b : Int) (nonzero : b ≠ 0) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    Verified a b heap globals := by
  simpa only [Verified, normalize_canonical a b nonzero] using result a b nonzero heap globals

end solvers.«93».rational
