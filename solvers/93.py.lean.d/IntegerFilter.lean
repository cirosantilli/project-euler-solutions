import solvers.«93.py.lean.d».Rational
import ProjectEulerSolutions.Lib.CanonicalIntegers
import lang.py.FormalSW.BooleanCalls

namespace solvers.«93».consecutive_length
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
set_option Elab.async false

abbrev program := Solvers.P93Ast.module
abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
abbrev denominatorTest : Expr := .compare (.name "denominator") [(.eq, .constant (.int 1))]
abbrev numeratorTest : Expr := .compare (.name "numerator") [(.gt, .constant (.int 0))]
abbrev emit : List Stmt := [.expr (.call (.attribute (.name "integers") "add") [.name "numerator"])]
abbrev filterBody : List Stmt := [.ifThen (.boolOp .and [denominatorTest, numeratorTest]) emit []]

theorem source_filter : ((findFunctionIn program.body "consecutive_length").getD ([], [])).2[1]? =
    some (.forIn (.tuple [.name "numerator", .name "denominator"])
      (.call (.name "all_results") [.name "digits"]) filterBody []) := rfl

/-- The denominator test precedes the numerator test, as in the Python short circuit. -/
theorem filter_test (pair : Pair) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getN : env "numerator" = some (number pair.1)) (getD : env "denominator" = some (number pair.2)) :
    EvaluatesTruth program env (.boolOp .and [denominatorTest, numeratorTest]) heap
      (decide (pair.2 = 1 ∧ 0 < pair.1)) globals := by
  have first : EvaluatesTruth program env denominatorTest heap (decide (pair.2 = 1)) globals := by
    refine ⟨.scalar (.bool (decide (pair.2 = 1))), ?_, rfl⟩
    exact Evaluates.compare (Evaluates.lookup getD) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  have second : EvaluatesTruth program env numeratorTest heap (decide (0 < pair.1)) globals := by
    refine ⟨.scalar (.bool (decide (0 < pair.1))), ?_, rfl⟩
    exact Evaluates.compare (Evaluates.lookup getN) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  simpa only [Bool.decide_and] using first.and second

def collect (heap : Mutable.Heap) (address : Nat) (items : List Constant) (pair : Pair) : Mutable.Heap :=
  if pair.2 = 1 ∧ 0 < pair.1 then heap.set address (.set (ScalarSet.insert items (.int pair.1))) else heap

/-- One source iteration inserts an accepted integer without reallocating the collector or copying tuples. -/
theorem filter_result (pair : Pair) (heap : Mutable.Heap) (address : Nat) (items : List Constant)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : heap[address]? = some (.set items))
    (getOut : env "integers" = some (.ref address))
    (getN : env "numerator" = some (number pair.1)) (getD : env "denominator" = some (number pair.2)) :
    BlockContinues program env filterBody heap env (collect heap address items pair) globals := by
  obtain ⟨test, evaluated, truth⟩ := filter_test pair heap env globals getN getD
  by_cases accepted : pair.2 = 1 ∧ 0 < pair.1
  · simp only [collect, if_pos accepted]
    apply BlockContinues.branch (flag := true) evaluated (by simpa [accepted] using truth)
    exact BlockContinues.discard (Evaluates.add (Evaluates.lookup getOut) found (Evaluates.lookup getN) found)
  · simp only [collect, if_neg accepted]
    exact BlockContinues.branch (flag := false) evaluated (by simpa [accepted] using truth) .nil

end solvers.«93».consecutive_length
