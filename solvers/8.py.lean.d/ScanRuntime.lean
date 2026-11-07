import solvers.«8.py.lean.d».Source
import lang.py.FormalSW.HeapRuntimeCostSequences

/-! Runtime bounds for both scans on arbitrary signed integer lists. -/
namespace solvers.«8».scanRuntime
open FormalSW.PyAst
open HeapCalls.Runtime
set_option maxRecDepth 4096
set_option maxHeartbeats 1200000
private abbrev program := Solvers.P8Ast.module

theorem read_digit (digits : List Int) (heap : Mutable.Heap) (address position : Nat)
    (env : HeapCalls.Env) (indexName : String)
    (found : heap[address]? = some (.list (ints digits))) (valid : position < digits.length)
    (receiver : env "digits" = some (.ref address))
    (index : env indexName = some (.scalar (.int position))) :
    EvaluatesIn .word program env (.subscript (.name "digits") (.name indexName)) heap 10
      (.scalar (.int digits[position])) heap := by
  apply EvaluatesIn.list_subscript (by intros; simp) (EvaluatesIn.lookup receiver) (EvaluatesIn.lookup index) found
  simpa [ints] using Mutable.getItem_list heap address position (ints digits) found (by simpa [ints] using valid)

private theorem scan_end (digits : List Int) (heap : Mutable.Heap) (position : Nat)
    (env : HeapCalls.Env) (name : String) (zeros : Bool)
    (index : env name = some (.scalar (.int position)))
    (count : env "digit_count" = some (.scalar (.int digits.length))) (finished : digits.length ≤ position) :
    EvaluatesIn .word program env (scanTest name zeros) heap 9 (.scalar (.bool false)) heap := by
  apply EvaluatesIn.boolean_stop (value := .bool false) (flag := false)
  · exact EvaluatesIn.compare (EvaluatesIn.lookup index) (EvaluatesIn.lookup count)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega) rfl
  · rfl
  · rfl

private theorem scan_read (digits : List Int) (heap : Mutable.Heap) (address position : Nat)
    (env : HeapCalls.Env) (name : String) (zeros : Bool)
    (found : heap[address]? = some (.list (ints digits))) (valid : position < digits.length)
    (receiver : env "digits" = some (.ref address))
    (index : env name = some (.scalar (.int position)))
    (count : env "digit_count" = some (.scalar (.int digits.length))) :
    EvaluatesIn .word program env (scanTest name zeros) heap 26
      (.scalar (.bool (decide (digits[position] = 0) == zeros))) heap := by
  apply EvaluatesIn.boolean_pair (ca := 7) (cb := 16) (value := .bool true)
      (flag := true) (answerFlag := decide (digits[position] = 0) == zeros)
  · exact EvaluatesIn.compare (EvaluatesIn.lookup index) (EvaluatesIn.lookup count)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, valid]) rfl
  · rfl
  · rfl
  · apply EvaluatesIn.compare (work := 2) (bv := .scalar (.int 0)) (read_digit digits heap address position env name found valid receiver index)
      EvaluatesIn.constant
    · cases zeros <;> simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · cases zeros <;> rfl
  · rfl

/-- Indexed reads, comparisons, increments and control cost at most 35 units
per consumed element, plus 28 to stop. The input heap is preserved. -/
theorem scan_bound (digits : List Int) (heap : Mutable.Heap) (address start stop : Nat)
    (env : HeapCalls.Env) (name : String) (zeros : Bool)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (count : env "digit_count" = some (.scalar (.int digits.length)))
    (notDigits : "digits" ≠ name) (notCount : "digit_count" ≠ name)
    (ordered : start ≤ stop) (bounded : stop ≤ digits.length)
    (selected : ∀ index, start ≤ index → (bound : index < stop) →
      (decide (digits[index]'(by omega) = 0) == zeros) = true)
    (finished : ∀ bound : stop < digits.length,
      (decide (digits[stop]'bound = 0) == zeros) = false) :
    ∃ cost, cost ≤ 35*(stop-start)+28 ∧
      BlockContinuesIn .word program (HeapCalls.setLocal env name (.scalar (.int start)))
        [scanLoop name zeros] heap cost (HeapCalls.setLocal env name (.scalar (.int stop))) heap := by
  by_cases atEnd : start = stop
  · subst start
    by_cases valid : stop < digits.length
    · refine ⟨28, by omega, ?_⟩
      apply BlockContinuesIn.while_loop (cc := 26) (cb := 0) (value := .bool false) (flag := false)
      · simpa only [finished valid] using scan_read digits heap address stop
          (HeapCalls.setLocal env name (.scalar (.int stop))) name zeros found valid
          (by simp [HeapCalls.setLocal, notDigits, receiver])
          (by simp [HeapCalls.setLocal]) (by simp [HeapCalls.setLocal, notCount, count])
      · rfl
      · exact BlockContinuesIn.nil
    · refine ⟨11, by omega, ?_⟩
      apply BlockContinuesIn.while_loop (cc := 9) (cb := 0) (value := .bool false) (flag := false)
      · exact scan_end digits heap stop _ name zeros
          (by simp [HeapCalls.setLocal]) (by simp [HeapCalls.setLocal, notCount, count]) (by omega)
      · rfl
      · exact BlockContinuesIn.nil
  · have before : start < stop := by omega
    obtain ⟨cost, bound, tail⟩ := scan_bound digits heap address (start+1) stop env name zeros found receiver count
      notDigits notCount (by omega) bounded (fun index lower upper => selected index (by omega) upper) finished
    refine ⟨26+(7+cost)+2, by omega, ?_⟩
    apply BlockContinuesIn.while_loop (cc := 26) (cb := 7+cost) (value := .bool true) (flag := true)
    · simpa only [selected start (by omega) before] using scan_read digits heap address start
        (HeapCalls.setLocal env name (.scalar (.int start))) name zeros found (by omega)
        (by simp [HeapCalls.setLocal, notDigits, receiver])
        (by simp [HeapCalls.setLocal]) (by simp [HeapCalls.setLocal, notCount, count])
    · rfl
    · apply BlockContinuesIn.append (middleEnv := HeapCalls.setLocal env name (.scalar (.int (start+1 : Nat))))
      · have increment := BlockContinuesIn.augAssign (arithmetic := .word) (module := program)
          (env := HeapCalls.setLocal env name (.scalar (.int start))) (name := name) (heap := heap)
          (op := .add) (left := .int start) (right := .int 1)
          (value := .scalar (.int (start+1 : Nat))) (by simp [HeapCalls.setLocal]) EvaluatesIn.constant
          (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary]) rfl
        simpa only [setLocal_shadow] using increment
      · exact tail
termination_by stop-start

end solvers.«8».scanRuntime
