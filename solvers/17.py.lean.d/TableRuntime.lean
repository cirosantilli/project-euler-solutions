import solvers.«17.py.lean.d».TableState
import lang.py.FormalSW.HeapRuntimeCostDictionaries
import lang.py.FormalSW.HeapRuntimeCostPrimitives
import lang.py.FormalSW.HeapRuntimeCostComparisons
import lang.py.FormalSW.HeapRuntimeCostScalarPairs

namespace solvers.«17».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open ProjectEulerStatements.P17
private abbrev program := Solvers.P17Ast.module

/-- Dictionary lookup scans at most ten integer keys; len reads the stored length. -/
theorem table_read (heap : Mutable.Heap) (globals : Mutable.Globals)
    (table : WordTable heap globals name lower upper length)
    (env : HeapCalls.Env) (index : String) (n : Nat)
    (getTable : env name = globals.lookup name)
    (getIndex : env index = some (.scalar (.int n)))
    (lo : lower ≤ n) (hi : n < upper) :
    EvaluatesIn .word program env
      (.builtinCall .len [.subscript (.name name) (.name index)]) heap 44
      (.scalar (.int (length n))) heap globals := by
  obtain ⟨address, entries, bound, found, count, integerKeys, words⟩ := table
  obtain ⟨word, present, size⟩ := words n lo hi
  have paid := dictionary_lookup_word_int_le_33 entries n count integerKeys
  apply EvaluatesIn.mono
  · apply EvaluatesIn.builtin_single (builtin := .len) (work := 1)
      (by intro h; cases h) (by intro h; cases h)
    · exact EvaluatesIn.dictionary_subscript (by intro a b c impossible; cases impossible)
        (.lookup (getTable.trans bound)) (.lookup getIndex) found present
    · simp [primitive, Mutable.length, size]
    · rfl
  · omega

/-- Inclusive source guard; its middle operand is evaluated only once. -/
theorem bounded_cost (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (index : String) (n lower upper : Int)
    (getN : env index = some (.scalar (.int n))) (lo : lower ≤ n) :
    EvaluatesIn .word program env
      (.compare (.constant (.int lower)) [(.ltE, .name index), (.ltE, .constant (.int upper))])
      heap 13 (.scalar (.bool (decide (n ≤ upper)))) heap globals := by
  apply EvaluatesIn.chain (ca := 1) (cb := 10) EvaluatesIn.constant
  apply EvaluatesIn.ComparesIn.next (ca := 1) (cb := 5) (work := 2)
    (comparisonResult := .bool true) (EvaluatesIn.lookup getN)
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, lo]
  · rfl
  · rfl
  · apply EvaluatesIn.ComparesIn.last (cost := 1) (work := 2) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl

theorem length_constant (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (word : String) :
    EvaluatesIn .word program env (.builtinCall .len [.constant (.string word)]) heap 5
      (.scalar (.int word.length)) heap globals := by
  exact EvaluatesIn.builtin_single (builtin := .len) (work := 1)
    (by intro h; cases h) (by intro h; cases h) EvaluatesIn.constant rfl rfl

end solvers.«17».runtimeCost
