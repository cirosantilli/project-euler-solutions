import «.formalsw».solvers.«17.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p17.py»
import FormalSW
import ProjectEulerStatements.Lib.NumberWords

namespace solvers.«17»
open FormalSW.PyAst
open ProjectEulerStatements.P17
private abbrev program := Solvers.P17Ast.module

/-- Only the observed keys and word lengths matter; other entries may be present. -/
private def WordTable (heap : Mutable.Heap) (globals : Mutable.Globals)
    (name : String) (lower upper : Nat) (length : Nat → Nat) : Prop :=
  ∃ address entries, globals.lookup name = some (.ref address) ∧
    heap[address]? = some (.dict entries) ∧
    ∀ n, lower ≤ n → n < upper → ∃ word : String,
      ScalarDict.lookup entries (.int n) = some (.scalar (.string word)) ∧
      word.length = length n

private theorem WordTable.append {heap : Mutable.Heap} {globals : Mutable.Globals}
    {name : String} {lower upper : Nat} {length : Nat → Nat}
    (table : WordTable heap globals name lower upper length) (extra : Mutable.Heap) :
    WordTable (heap ++ extra) globals name lower upper length := by
  obtain ⟨address, entries, bound, found, words⟩ := table
  refine ⟨address, entries, bound, ?_, words⟩
  rw [List.getElem?_append_left]
  · exact found
  · by_contra outside
    simp [List.getElem?_eq_none (show heap.length ≤ address by omega)] at found

private theorem WordTable.read {heap : Mutable.Heap} {globals : Mutable.Globals}
    {name : String} {lower upper : Nat} {length : Nat → Nat}
    (table : WordTable heap globals name lower upper length)
    (env : HeapCalls.Env) (index : String) (n : Nat)
    (getTable : env name = globals.lookup name)
    (getIndex : env index = some (.scalar (.int n)))
    (lo : lower ≤ n) (hi : n < upper) :
    HeapCalls.Evaluates program env
      (.builtinCall .len [.subscript (.name name) (.name index)]) heap
      (.scalar (.int (length n))) heap globals := by
  obtain ⟨address, entries, bound, found, words⟩ := table
  obtain ⟨word, present, size⟩ := words n lo hi
  apply HeapCalls.Evaluates.builtin (values := [.scalar (.string word)]) (middle := heap)
  · exact .cons (HeapCalls.Evaluates.subscript
      (by intro a b c impossible; cases impossible)
      (.lookup (getTable.trans bound)) (.lookup getIndex)
      (Mutable.getItem_dictionary heap address entries found (.int n) present)) .nil
  · simp [HeapCalls.primitive, Mutable.length, size]

private def Tables (heap : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  WordTable heap globals "ONES" 0 10 smallWordLen ∧
  WordTable heap globals "TEENS" 10 20 smallWordLen ∧
  WordTable heap globals "TENS" 2 10 tensWordLen

private theorem Tables.append {heap : Mutable.Heap} {globals : Mutable.Globals}
    (tables : Tables heap globals) (extra : Mutable.Heap) :
    Tables (heap ++ extra) globals :=
  ⟨tables.1.append extra, tables.2.1.append extra, tables.2.2.append extra⟩

private theorem bounded (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (index : String) (n lower upper : Int)
    (getN : env index = some (.scalar (.int n))) (lo : lower ≤ n) :
    HeapCalls.Evaluates program env
      (.compare (.constant (.int lower)) [(.ltE, .name index), (.ltE, .constant (.int upper))])
      heap (.scalar (.bool (decide (n ≤ upper)))) heap globals := by
  apply HeapCalls.Evaluates.chain HeapCalls.Evaluates.constant
  apply HeapCalls.Evaluates.Compares.next (compared := .scalar (.bool true))
    (afterComparison := heap) (HeapCalls.Evaluates.lookup getN)
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, lo]
  · rfl
  · exact HeapCalls.Evaluates.Compares.last .constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

private abbrev belowHundredBody : List Stmt :=
  executableBody ((findFunctionIn program.body "letters_1_to_99").getD ([], [])).2

private theorem letters_1_to_99.block (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tables : Tables heap globals) (env : HeapCalls.Env) (n : Nat)
    (lo : 1 ≤ n) (hi : n ≤ 99)
    (getN : env "n" = some (.scalar (.int n)))
    (getOnes : env "ONES" = globals.lookup "ONES")
    (getTeens : env "TEENS" = globals.lookup "TEENS")
    (getTens : env "TENS" = globals.lookup "TENS") :
    ∃ extra, HeapCalls.BlockReturns program env belowHundredBody heap
      (.scalar (.int (wordLen n))) (heap ++ extra) globals := by
  have guard := bounded env heap globals "n" n 1 99 getN (by omega)
  simp only [show (n : Int) ≤ 99 by omega, decide_true] at guard
  by_cases ones : n < 10
  · refine ⟨[], ?_⟩
    simp only [List.append_nil]
    apply HeapCalls.BlockReturns.assertion guard rfl
    apply HeapCalls.BlockReturns.branch (flag := true)
      (HeapCalls.Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap)
        (.lookup getN) .constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, ones])) rfl
    rw [wordLen_below_twenty n (by omega)]
    exact HeapCalls.BlockReturns.return_value (tables.1.read env "n" n getOnes getN (by omega) ones)
  · have teenGuard := bounded env heap globals "n" n 10 19 getN (by omega)
    by_cases teens : n < 20
    · refine ⟨[], ?_⟩
      simp only [List.append_nil]
      apply HeapCalls.BlockReturns.assertion guard rfl
      apply HeapCalls.BlockReturns.branch (flag := false)
        (HeapCalls.Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap)
          (.lookup getN) .constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, ones])) rfl
      apply HeapCalls.BlockReturns.branch (flag := true)
        (by simpa only [show (n : Int) ≤ 19 by omega, decide_true] using teenGuard) rfl
      rw [wordLen_below_twenty n teens]
      exact HeapCalls.BlockReturns.return_value
        (tables.2.1.read env "n" n getTeens getN (by omega) teens)
    · let pair : Mutable.Object := .tuple
        [.scalar (.int ((n / 10 : Nat) : Int)), .scalar (.int ((n % 10 : Nat) : Int))]
      refine ⟨[pair], ?_⟩
      apply HeapCalls.BlockReturns.assertion guard rfl
      apply HeapCalls.BlockReturns.branch (flag := false)
        (HeapCalls.Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap)
          (.lookup getN) .constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, ones])) rfl
      apply HeapCalls.BlockReturns.branch (flag := false)
        (by simpa only [show ¬ (n : Int) ≤ 19 by omega, decide_false] using teenGuard) rfl
      apply HeapCalls.BlockReturns.unpack
        (items := [.scalar (.int ((n / 10 : Nat) : Int)), .scalar (.int ((n % 10 : Nat) : Int))])
        (afterUnpack := heap ++ [pair]) rfl
      · apply HeapCalls.Evaluates.builtin (.cons (.lookup getN) (.cons .constant .nil))
        have divided := Mutable.divmod_ints (n : Int) 10 (by decide) heap
        have quotient : (n : Int).fdiv 10 = ((n / 10 : Nat) : Int) := by
          rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide)]; rfl
        have remaining : (n : Int) - 10 * (n : Int).fdiv 10 = ((n % 10 : Nat) : Int) := by
          rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide), ← Int.emod_def]; rfl
        simpa only [quotient, remaining] using divided
      · simp [Mutable.sequence, pair, Int.emod_def]
      · rfl
      · rw [wordLen_below_hundred n (by omega), if_neg teens]
        apply HeapCalls.BlockReturns.return_value
        apply HeapCalls.Evaluates.binary
        · exact (tables.2.2.append [pair]).read _ "t" (n / 10)
            (by simpa [HeapCalls.assignNames, HeapCalls.setLocal] using getTens)
            (by simp [HeapCalls.assignNames, HeapCalls.setLocal]) (by omega) (by omega)
        · exact (tables.1.append [pair]).read _ "u" (n % 10)
            (by simpa [HeapCalls.assignNames, HeapCalls.setLocal] using getOnes)
            (by simp [HeapCalls.assignNames, HeapCalls.setLocal]) (by omega) (by omega)
        · simp [Mutable.binary, scalarBinary]

private theorem letters_1_to_99.result (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tables : Tables heap globals) (n : Nat) (lo : 1 ≤ n) (hi : n ≤ 99) :
    ∃ extra, HeapCalls.ReturnsValue program "letters_1_to_99" [.scalar (.int n)] heap
      (.scalar (.int (wordLen n))) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "letters_1_to_99").getD ([], [])).1
  let body := ((findFunctionIn program.body "letters_1_to_99").getD ([], [])).2
  obtain ⟨extra, executed⟩ := letters_1_to_99.block heap globals tables
    (HeapCalls.functionEnv args body [.scalar (.int n)] globals) n lo hi
    (by cases globals <;> rfl)
    (by cases globals with
        | nil => rfl
        | cons entry rest =>
            change (if "ONES" ∈ blockLocals body then none else _) = _
            rw [if_neg (by
              simp [body, program, Solvers.P17Ast.module, findFunctionIn,
                blockLocals, assignedLocals, targetLocals])])
    (by cases globals with
        | nil => rfl
        | cons entry rest =>
            change (if "TEENS" ∈ blockLocals body then none else _) = _
            rw [if_neg (by
              simp [body, program, Solvers.P17Ast.module, findFunctionIn,
                blockLocals, assignedLocals, targetLocals])])
    (by cases globals with
        | nil => rfl
        | cons entry rest =>
            change (if "TENS" ∈ blockLocals body then none else _) = _
            rw [if_neg (by
              simp [body, program, Solvers.P17Ast.module, findFunctionIn,
                blockLocals, assignedLocals, targetLocals])])
  exact ⟨extra, HeapCalls.BlockReturns.function rfl rfl executed⟩

private abbrev numberBody : List Stmt :=
  executableBody ((findFunctionIn program.body "letters_in_number").getD ([], [])).2

private theorem letters_in_number.hundreds (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tables : Tables heap globals) (env : HeapCalls.Env) (n : Nat)
    (lo : 100 ≤ n) (hi : n < 1000)
    (getN : env "n" = some (.scalar (.int n)))
    (getOnes : env "ONES" = globals.lookup "ONES") :
    ∃ extra, HeapCalls.BlockReturns program env (numberBody.drop 3) heap
      (.scalar (.int (wordLen n))) (heap ++ extra) globals := by
  let pair : Mutable.Object := .tuple
    [.scalar (.int ((n / 100 : Nat) : Int)), .scalar (.int ((n % 100 : Nat) : Int))]
  let unpacked := HeapCalls.assignNames env ["h", "r"]
    [.scalar (.int ((n / 100 : Nat) : Int)), .scalar (.int ((n % 100 : Nat) : Int))]
  let total := smallWordLen (n / 100) + 7
  let ready := HeapCalls.setLocal unpacked "total" (.scalar (.int total))
  have getR : ready "r" = some (.scalar (.int ((n % 100 : Nat) : Int))) := by
    simp [ready, unpacked, HeapCalls.assignNames, HeapCalls.setLocal]
  have finish : ∃ extra, HeapCalls.BlockReturns program ready (numberBody.drop 5)
      (heap ++ [pair]) (.scalar (.int (wordLen n))) ((heap ++ [pair]) ++ extra) globals := by
    rw [wordLen_hundreds n lo hi]
    by_cases zero : n % 100 = 0
    · refine ⟨[], ?_⟩
      simp only [if_pos zero, Nat.add_zero, List.append_nil]
      apply HeapCalls.BlockReturns.branch (flag := false)
        (HeapCalls.Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap ++ [pair])
          (.lookup getR) .constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero])) rfl
      exact HeapCalls.BlockReturns.return_value (.lookup (by simp [ready, HeapCalls.setLocal, total]))
    · obtain ⟨extra, called⟩ := letters_1_to_99.result (heap ++ [pair]) globals (tables.append [pair])
        (n % 100) (by omega) (by omega)
      refine ⟨extra, ?_⟩
      rw [if_neg zero]
      apply HeapCalls.BlockReturns.branch (flag := true)
        (HeapCalls.Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap ++ [pair])
          (.lookup getR) .constant
          (by
            simp [HeapCalls.compare, scalarCompare, evalCmpOp]
            exact_mod_cast (Nat.dvd_iff_mod_eq_zero.not.mpr zero))) rfl
      apply HeapCalls.BlockReturns.augAssign (left := .scalar (.int total))
        (right := .scalar (.int 3)) (value := .scalar (.int (total + 3)))
        (middle := heap ++ [pair]) (afterOperation := heap ++ [pair])
        (by simp [ready, HeapCalls.setLocal])
      · exact HeapCalls.Evaluates.builtin (.cons .constant .nil) rfl
      · rfl
      · apply HeapCalls.BlockReturns.branch (flag := true)
          (HeapCalls.Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap ++ [pair])
            (.lookup (by simpa [HeapCalls.setLocal] using getR)) .constant
            (by
              simp [HeapCalls.compare, scalarCompare, evalCmpOp]
              omega)) rfl
        apply HeapCalls.BlockReturns.augAssign
          (left := .scalar (.int (total + 3)))
          (right := .scalar (.int (wordLen (n % 100))))
          (value := .scalar (.int ((total + 3 : Nat) + wordLen (n % 100))))
          (middle := (heap ++ [pair]) ++ extra) (afterOperation := (heap ++ [pair]) ++ extra)
          (by simp [HeapCalls.setLocal])
        · exact HeapCalls.Evaluates.call
            (.cons (.lookup (by simpa [HeapCalls.setLocal] using getR)) .nil) called
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
        · apply HeapCalls.BlockReturns.return_value
          apply HeapCalls.Evaluates.lookup
          simp [HeapCalls.setLocal, total]
          omega
  obtain ⟨extra, finish⟩ := finish
  refine ⟨[pair] ++ extra, ?_⟩
  apply HeapCalls.BlockReturns.unpack
    (items := [.scalar (.int ((n / 100 : Nat) : Int)), .scalar (.int ((n % 100 : Nat) : Int))])
    (afterUnpack := heap ++ [pair]) rfl
  · apply HeapCalls.Evaluates.builtin (.cons (.lookup getN) (.cons .constant .nil))
    have divided := Mutable.divmod_ints (n : Int) 100 (by decide) heap
    have quotient : (n : Int).fdiv 100 = ((n / 100 : Nat) : Int) := by
      rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide)]; rfl
    have remaining : (n : Int) - 100 * (n : Int).fdiv 100 = ((n % 100 : Nat) : Int) := by
      rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide), ← Int.emod_def]; rfl
    simpa only [quotient, remaining] using divided
  · simp [Mutable.sequence, pair, Int.emod_def]
  · rfl
  · apply HeapCalls.BlockReturns.assign (value := .scalar (.int total)) (middle := heap ++ [pair])
    · apply HeapCalls.Evaluates.binary
      · exact (tables.1.append [pair]).read _ "h" (n / 100)
          (by simpa [HeapCalls.assignNames, HeapCalls.setLocal] using getOnes)
          (by simp [HeapCalls.assignNames, HeapCalls.setLocal]) (by omega) (by omega)
      · exact HeapCalls.Evaluates.builtin (.cons .constant .nil) rfl
      · simp [total, Mutable.binary, scalarBinary]
        decide
    · simpa only [ready, unpacked, List.append_assoc] using finish

private theorem letters_in_number.block (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tables : Tables heap globals) (env : HeapCalls.Env) (n : Nat)
    (lo : 1 ≤ n) (hi : n ≤ 1000)
    (getN : env "n" = some (.scalar (.int n)))
    (getOnes : env "ONES" = globals.lookup "ONES") :
    ∃ extra, HeapCalls.BlockReturns program env numberBody heap
      (.scalar (.int (wordLen n))) (heap ++ extra) globals := by
  have guard := bounded env heap globals "n" n 1 1000 getN (by omega)
  simp only [show (n : Int) ≤ 1000 by omega, decide_true] at guard
  by_cases thousand : n = 1000
  · subst n
    refine ⟨[], ?_⟩
    simp only [List.append_nil, wordLen_thousand]
    apply HeapCalls.BlockReturns.assertion guard rfl
    apply HeapCalls.BlockReturns.branch (flag := true)
      (HeapCalls.Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap)
        (.lookup getN) .constant rfl) rfl
    apply HeapCalls.BlockReturns.return_value
    exact HeapCalls.Evaluates.binary
      (HeapCalls.Evaluates.builtin (.cons .constant .nil) rfl)
      (HeapCalls.Evaluates.builtin (.cons .constant .nil) rfl) rfl
  · by_cases small : n < 100
    · obtain ⟨extra, called⟩ := letters_1_to_99.result heap globals tables n lo (by omega)
      refine ⟨extra, ?_⟩
      apply HeapCalls.BlockReturns.assertion guard rfl
      apply HeapCalls.BlockReturns.branch (flag := false)
        (HeapCalls.Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap)
          (.lookup getN) .constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, show (n : Int) ≠ 1000 by omega])) rfl
      apply HeapCalls.BlockReturns.branch (flag := true)
        (HeapCalls.Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap)
          (.lookup getN) .constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])) rfl
      exact HeapCalls.BlockReturns.return_value (HeapCalls.Evaluates.call (.cons (.lookup getN) .nil) called)
    · obtain ⟨extra, tail⟩ := letters_in_number.hundreds heap globals tables env n
        (by omega) (by omega) getN getOnes
      refine ⟨extra, ?_⟩
      apply HeapCalls.BlockReturns.assertion guard rfl
      apply HeapCalls.BlockReturns.branch (flag := false)
        (HeapCalls.Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap)
          (.lookup getN) .constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, show (n : Int) ≠ 1000 by omega])) rfl
      apply HeapCalls.BlockReturns.branch (flag := false)
        (HeapCalls.Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap)
          (.lookup getN) .constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])) rfl
      exact tail

private theorem letters_in_number.result (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tables : Tables heap globals) (n : Nat) (lo : 1 ≤ n) (hi : n ≤ 1000) :
    ∃ extra, HeapCalls.ReturnsValue program "letters_in_number" [.scalar (.int n)] heap
      (.scalar (.int (wordLen n))) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "letters_in_number").getD ([], [])).1
  let body := ((findFunctionIn program.body "letters_in_number").getD ([], [])).2
  obtain ⟨extra, executed⟩ := letters_in_number.block heap globals tables
    (HeapCalls.functionEnv args body [.scalar (.int n)] globals) n lo hi
    (by cases globals <;> rfl)
    (by cases globals with
        | nil => rfl
        | cons entry rest =>
            change (if "ONES" ∈ blockLocals body then none else _) = _
            rw [if_neg (by
              simp [body, program, Solvers.P17Ast.module, findFunctionIn,
                blockLocals, assignedLocals, targetLocals])])
  exact ⟨extra, HeapCalls.BlockReturns.function rfl rfl executed⟩

private abbrev sumElement : Expr := .call (.name "letters_in_number") [.name "i"]
private abbrev sumBody : List Stmt := HeapCalls.sumBody sumElement []

/-- Every helper call appends private tuples, preserving the shared word tables. -/
private theorem solve.loop (count first : Nat) (initial : Int) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (tables : Tables heap globals)
    (lo : 1 ≤ first) (hi : first + count ≤ 1001)
    (env : HeapCalls.Env) (accumulator : env HeapCalls.sumAccumulator = some (.scalar (.int initial))) :
    ∃ extra, HeapCalls.BlockReturns program env
      [.rangeStepLoop "i" (first : Int) ((first + count : Nat) : Int) 1 sumBody [],
        .return (.name HeapCalls.sumAccumulator)] heap
      (.scalar (.int (initial + ((List.range' first count).map
        (fun i => (wordLen i : Int))).sum))) (heap ++ extra) globals := by
  induction count generalizing first initial heap env with
  | zero =>
      refine ⟨[], ?_⟩
      simp only [Nat.add_zero, List.range'_zero, List.map_nil, List.sum_nil, Int.add_zero, List.append_nil]
      apply HeapCalls.BlockReturns.range_done (by simp [Calls.rangeContinues])
      exact HeapCalls.BlockReturns.return_value (.lookup accumulator)
  | succ count ih =>
      let withI := HeapCalls.setLocal env "i" (.scalar (.int first))
      let nextEnv := HeapCalls.setLocal withI HeapCalls.sumAccumulator
        (.scalar (.int (initial + wordLen first)))
      obtain ⟨allocated, called⟩ := letters_in_number.result heap globals tables first lo (by omega)
      obtain ⟨later, continued⟩ := ih (first + 1) (initial + wordLen first)
        (heap ++ allocated) (tables.append allocated) (by omega) (by omega)
        nextEnv (by simp [nextEnv, HeapCalls.setLocal])
      refine ⟨allocated ++ later, ?_⟩
      apply HeapCalls.BlockReturns.range_step (by simp [Calls.rangeContinues])
      apply HeapCalls.BlockReturns.assign (value := .scalar (.int (initial + wordLen first)))
        (middle := heap ++ allocated)
      · apply HeapCalls.Evaluates.binary
          (.lookup (by simpa [HeapCalls.setLocal, HeapCalls.sumAccumulator] using accumulator))
          (HeapCalls.Evaluates.call (.cons (.lookup (by simp [HeapCalls.setLocal])) .nil) called)
        rfl
      · change HeapCalls.BlockReturns program nextEnv
          [.rangeStepLoop "i" ((first : Int) + 1) ((first + (count + 1) : Nat) : Int) 1 sumBody [],
            .return (.name HeapCalls.sumAccumulator)] (heap ++ allocated) _ _ globals
        simpa only [List.range'_succ, List.map_cons, List.sum_cons, Nat.cast_add, Nat.cast_one,
          Nat.add_assoc, Nat.add_left_comm, Nat.add_comm, Int.add_assoc, Int.add_left_comm, Int.add_comm,
          List.append_assoc] using continued

private theorem solve.result (n : Nat) (bound : n ≤ 1000)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (tables : Tables heap globals) :
    ∃ extra, HeapCalls.ReturnsValue program "solve" [.scalar (.int n)] heap
      (.scalar (.int (naive n))) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let env := HeapCalls.functionEnv args body [.scalar (.int n)] globals
  let summed := HeapCalls.setLocal (HeapCalls.comprehensionEnv env
    [("i", .call (.name "range") [.constant (.int 1), .binOp (.name "n") .add (.constant (.int 1))], [])])
    HeapCalls.sumAccumulator (.scalar (.int 0))
  obtain ⟨extra, loop⟩ := solve.loop n 1 0 heap globals tables (by omega) (by omega)
    summed (by simp [summed, HeapCalls.setLocal])
  have total : ((List.range' 1 n).map (fun i => (wordLen i : Int))).sum = (naive n : Int) := by
    simpa using congrArg (fun x : Nat => (x : Int)) (sum_wordLen n)
  refine ⟨extra, ?_⟩
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.return_value
  apply HeapCalls.Evaluates.sum_range
  apply HeapCalls.Evaluates.sum_generator_range
    (.cons .constant (.cons
      (HeapCalls.Evaluates.binary (.lookup (by cases globals <;> rfl)) .constant rfl) .nil))
    rfl HeapCalls.Evaluates.constant rfl
  simpa only [Int.zero_add, total, Nat.cast_add, Nat.cast_one, Int.add_comm] using loop

private theorem equal_string_entry (word : String) (value : Mutable.Value) :
    HeapCalls.equalElements [.scalar (.string word)] [value] =
      some (decide (value = .scalar (.string word))) := by
  cases value with
  | ref address => rfl
  | scalar constant =>
      cases constant <;> simp [HeapCalls.equalElements, scalarCompare, eq_comm]
      split <;> simp_all

private theorem equal_string_entries (expected : List (Constant × String))
    (entries : List (Constant × Mutable.Value)) :
    HeapCalls.equalEntries (expected.map fun (key, word) => (key, .scalar (.string word))) entries =
      some (expected.all fun (key, word) => decide
        (ScalarDict.lookup entries key = some (.scalar (.string word)))) := by
  induction expected with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨key, word⟩ := entry
      cases found : ScalarDict.lookup entries key with
      | none => simp [HeapCalls.equalEntries, found]
      | some value =>
          simp only [List.map_cons, HeapCalls.equalEntries, found, equal_string_entry]
          by_cases same : value = .scalar (.string word)
          · simp [same, ih, found]
          · simp [same, found]


private def wordEntries (expected : List (Constant × String)) : List (Constant × Mutable.Value) :=
  expected.map fun (key, word) => (key, .scalar (.string word))

private def tableMatches (expected : List (Constant × String))
    (entries : List (Constant × Mutable.Value)) : Bool :=
  decide (expected.length = entries.length) &&
    expected.all fun (key, word) => decide
      (ScalarDict.lookup entries key = some (.scalar (.string word)))

private theorem table_test (expected : List (Constant × String))
    (heap : Mutable.Heap) (globals : Mutable.Globals) (env : HeapCalls.Env)
    (literal : Expr) (name : String) (address : Nat) (entries : List (Constant × Mutable.Value))
    (constructed : HeapCalls.Evaluates program env literal heap (.ref heap.length)
      (heap ++ [.dict (wordEntries expected)]) globals)
    (getTable : env name = some (.ref address)) (found : heap[address]? = some (.dict entries)) :
    HeapCalls.Evaluates program env (.compare literal [(.eq, .name name)]) heap
      (.scalar (.bool (tableMatches expected entries))) (heap ++ [.dict (wordEntries expected)]) globals := by
  have inside : address < heap.length := by
    by_contra outside
    simp [List.getElem?_eq_none (show heap.length ≤ address by omega)] at found
  have preserved : (heap ++ [Mutable.Object.dict (wordEntries expected)])[address]? = some (.dict entries) := by
    rw [List.getElem?_append_left inside]
    exact found
  apply HeapCalls.Evaluates.compare constructed (.lookup getTable)
  simp [HeapCalls.compare, preserved, show heap.length ≠ address by omega]
  simp [HeapCalls.equalDictionaries, wordEntries, equal_string_entries, tableMatches]
  split <;> simp_all <;> rfl

private def onesExpected : List (Constant × String) :=
  [(.int 0, ""), (.int 1, "one"), (.int 2, "two"), (.int 3, "three"), (.int 4, "four"), (.int 5, "five"), (.int 6, "six"), (.int 7, "seven"), (.int 8, "eight"), (.int 9, "nine")]

private theorem ones_test (heap : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env) (address : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : env "ONES" = some (.ref address)) (found : heap[address]? = some (.dict entries)) :
    HeapCalls.Evaluates program env Solvers.P17Ast.requires_solve_1 heap
      (.scalar (.bool (tableMatches onesExpected entries)))
      (heap ++ [.dict (wordEntries onesExpected)]) globals := by
  apply table_test onesExpected heap globals env _ "ONES" address entries _ getTable found
  exact HeapCalls.Evaluates.dictionary_constants (onesExpected.map fun (key, word) => (key, .string word))

private theorem ones_of_condition (heap : Mutable.Heap) (globals : Mutable.Globals)
    (address : Nat) (bound : globals.lookup "ONES" = some (.ref address))
    (typed : Mutable.Value.HasType heap (.ref address) "dict[int,str]")
    (condition : HeapCalls.Condition program (fun name => globals.lookup name)
      Solvers.P17Ast.requires_solve_1 heap globals) :
    WordTable heap globals "ONES" 0 10 smallWordLen := by
  obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  have executed := ones_test heap globals (fun name => globals.lookup name) address entries bound found
  have matched : tableMatches onesExpected entries = true := by
    simpa using executed.condition_iff.mp condition
  have keys := List.all_eq_true.mp (Bool.and_eq_true_iff.mp matched).2
  refine ⟨address, entries, bound, found, ?_⟩
  intro n lo hi
  refine ⟨onesWords[n - 0]!, ?_, ?_⟩
  · exact of_decide_eq_true (keys (.int n, onesWords[n - 0]!) (by interval_cases n <;> simp [onesExpected, onesWords]))
  · simpa using onesWords_length n hi

private def teensExpected : List (Constant × String) :=
  [(.int 10, "ten"), (.int 11, "eleven"), (.int 12, "twelve"), (.int 13, "thirteen"), (.int 14, "fourteen"), (.int 15, "fifteen"), (.int 16, "sixteen"), (.int 17, "seventeen"), (.int 18, "eighteen"), (.int 19, "nineteen")]

private theorem teens_test (heap : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env) (address : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : env "TEENS" = some (.ref address)) (found : heap[address]? = some (.dict entries)) :
    HeapCalls.Evaluates program env Solvers.P17Ast.requires_solve_2 heap
      (.scalar (.bool (tableMatches teensExpected entries)))
      (heap ++ [.dict (wordEntries teensExpected)]) globals := by
  apply table_test teensExpected heap globals env _ "TEENS" address entries _ getTable found
  exact HeapCalls.Evaluates.dictionary_constants (teensExpected.map fun (key, word) => (key, .string word))

private theorem teens_of_condition (heap : Mutable.Heap) (globals : Mutable.Globals)
    (address : Nat) (bound : globals.lookup "TEENS" = some (.ref address))
    (typed : Mutable.Value.HasType heap (.ref address) "dict[int,str]")
    (condition : HeapCalls.Condition program (fun name => globals.lookup name)
      Solvers.P17Ast.requires_solve_2 heap globals) :
    WordTable heap globals "TEENS" 10 20 smallWordLen := by
  obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  have executed := teens_test heap globals (fun name => globals.lookup name) address entries bound found
  have matched : tableMatches teensExpected entries = true := by
    simpa using executed.condition_iff.mp condition
  have keys := List.all_eq_true.mp (Bool.and_eq_true_iff.mp matched).2
  refine ⟨address, entries, bound, found, ?_⟩
  intro n lo hi
  refine ⟨teenWords[n - 10]!, ?_, ?_⟩
  · exact of_decide_eq_true (keys (.int n, teenWords[n - 10]!) (by interval_cases n <;> simp [teensExpected, teenWords]))
  · simpa using teenWords_length n lo hi

private def tensExpected : List (Constant × String) :=
  [(.int 2, "twenty"), (.int 3, "thirty"), (.int 4, "forty"), (.int 5, "fifty"), (.int 6, "sixty"), (.int 7, "seventy"), (.int 8, "eighty"), (.int 9, "ninety")]

private theorem tens_test (heap : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env) (address : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : env "TENS" = some (.ref address)) (found : heap[address]? = some (.dict entries)) :
    HeapCalls.Evaluates program env Solvers.P17Ast.requires_solve_3 heap
      (.scalar (.bool (tableMatches tensExpected entries)))
      (heap ++ [.dict (wordEntries tensExpected)]) globals := by
  apply table_test tensExpected heap globals env _ "TENS" address entries _ getTable found
  exact HeapCalls.Evaluates.dictionary_constants (tensExpected.map fun (key, word) => (key, .string word))

private theorem tens_of_condition (heap : Mutable.Heap) (globals : Mutable.Globals)
    (address : Nat) (bound : globals.lookup "TENS" = some (.ref address))
    (typed : Mutable.Value.HasType heap (.ref address) "dict[int,str]")
    (condition : HeapCalls.Condition program (fun name => globals.lookup name)
      Solvers.P17Ast.requires_solve_3 heap globals) :
    WordTable heap globals "TENS" 2 10 tensWordLen := by
  obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  have executed := tens_test heap globals (fun name => globals.lookup name) address entries bound found
  have matched : tableMatches tensExpected entries = true := by
    simpa using executed.condition_iff.mp condition
  have keys := List.all_eq_true.mp (Bool.and_eq_true_iff.mp matched).2
  refine ⟨address, entries, bound, found, ?_⟩
  intro n lo hi
  refine ⟨tensWords[n - 2]!, ?_, ?_⟩
  · exact of_decide_eq_true (keys (.int n, tensWords[n - 2]!) (by interval_cases n <;> simp [tensExpected, tensWords]))
  · simpa using tensWords_length n lo hi

private abbrev context (ones teens tens : Nat) : Mutable.Globals :=
  [("ONES", .ref ones), ("TEENS", .ref teens), ("TENS", .ref tens)]

private abbrev conditionEnv (n : Int) (ones teens tens : Nat) : HeapCalls.Env :=
  fun name => if name = "n" then some (.scalar (.int n)) else (context ones teens tens).lookup name

private abbrev boundCondition (heap : Mutable.Heap) (n : Int) (ones teens tens : Nat) : Prop :=
  HeapCalls.Condition program (conditionEnv n ones teens tens) Solvers.P17Ast.requires_solve_0
    heap (context ones teens tens)

private abbrev tableCondition (expression : Expr) (heap : Mutable.Heap) (ones teens tens : Nat) : Prop :=
  HeapCalls.Condition program (fun name => (context ones teens tens).lookup name) expression
    heap (context ones teens tens)

private theorem bound_test (heap : Mutable.Heap) (n : Int) (ones teens tens : Nat) :
    HeapCalls.Evaluates program (conditionEnv n ones teens tens) Solvers.P17Ast.requires_solve_0 heap
      (.scalar (.bool (decide (0 ≤ n ∧ n ≤ 1000)))) heap (context ones teens tens) := by
  by_cases nonnegative : 0 ≤ n
  · simpa only [nonnegative, true_and] using
      bounded (conditionEnv n ones teens tens) heap (context ones teens tens) "n" n 0 1000 rfl nonnegative
  · simp only [nonnegative, false_and, decide_false]
    apply HeapCalls.Evaluates.chain .constant
    exact HeapCalls.Evaluates.Compares.stop (.lookup (show conditionEnv n ones teens tens "n" = _ from rfl))
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, nonnegative]) rfl

private theorem tables_of_conditions (heap : Mutable.Heap) (ones teens tens : Nat)
    (onesTyped : Mutable.Value.HasType heap (.ref ones) "dict[int,str]")
    (teensTyped : Mutable.Value.HasType heap (.ref teens) "dict[int,str]")
    (tensTyped : Mutable.Value.HasType heap (.ref tens) "dict[int,str]")
    (onesCondition : tableCondition Solvers.P17Ast.requires_solve_1 heap ones teens tens)
    (teensCondition : tableCondition Solvers.P17Ast.requires_solve_2 heap ones teens tens)
    (tensCondition : tableCondition Solvers.P17Ast.requires_solve_3 heap ones teens tens) :
    Tables heap (context ones teens tens) :=
  ⟨ones_of_condition heap _ ones rfl onesTyped onesCondition,
   teens_of_condition heap _ teens rfl teensTyped teensCondition,
   tens_of_condition heap _ tens rfl tensTyped tensCondition⟩

private abbrev Guarded (property : Mutable.Heap → Int → Nat → Nat → Nat → Prop) : Prop :=
  ∀ heap n ones teens tens, heap.Valid →
    Mutable.Value.HasType heap (.ref ones) "dict[int,str]" →
    Mutable.Value.HasType heap (.ref teens) "dict[int,str]" →
    Mutable.Value.HasType heap (.ref tens) "dict[int,str]" →
    boundCondition heap n ones teens tens →
    tableCondition Solvers.P17Ast.requires_solve_1 heap ones teens tens →
    tableCondition Solvers.P17Ast.requires_solve_2 heap ones teens tens →
    tableCondition Solvers.P17Ast.requires_solve_3 heap ones teens tens →
    property heap n ones teens tens

theorem solve.«Equiv(naive)» : Guarded (fun heap n ones teens tens =>
    (HeapCalls.call program "solve" [.scalar (.int n)] heap (context ones teens tens)).Equiv
      (HeapCalls.call P17Ast.module "naive" [.scalar (.int n)] heap (context ones teens tens))) := by
  intro heap n ones teens tens valid onesTyped teensTyped tensTyped bound onesCondition teensCondition tensCondition
  have bounds : 0 ≤ n ∧ n ≤ 1000 := by
    simpa using (bound_test heap n ones teens tens).condition_iff.mp bound
  lift n to Nat using bounds.1
  have tables := tables_of_conditions heap ones teens tens onesTyped teensTyped tensTyped
    onesCondition teensCondition tensCondition
  obtain ⟨extra, optimized⟩ := solve.result n (by omega) heap (context ones teens tens) tables
  obtain ⟨referenceExtra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p17.naive.result
      (globals := context ones teens tens) n heap
  apply Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
  apply Mutable.Completion.equivalent_allocations heap extra referenceExtra
    (.scalar (.int (naive n))) (context ones teens tens) valid trivial
  have inside (address : Nat) (typed : Mutable.Value.HasType heap (.ref address) "dict[int,str]") :
      address < heap.length := by
    obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType typed rfl
    by_contra outside
    simp [List.getElem?_eq_none (show heap.length ≤ address by omega)] at found
  intro entry member
  simp only [context, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact inside ones onesTyped
  · exact inside teens teensTyped
  · exact inside tens tensTyped

theorem solve.«Supported» :
    Guarded (fun heap n ones teens tens =>
      (HeapCalls.call program "solve" [.scalar (.int n)] heap (context ones teens tens)).Supported) ∧
    Guarded (fun heap n ones teens tens =>
      (HeapCalls.call P17Ast.module "naive" [.scalar (.int n)] heap (context ones teens tens)).Supported) := by
  constructor
  · intro heap n ones teens tens _valid onesTyped teensTyped tensTyped bound onesCondition teensCondition tensCondition
    have bounds : 0 ≤ n ∧ n ≤ 1000 := by
      simpa using (bound_test heap n ones teens tens).condition_iff.mp bound
    lift n to Nat using bounds.1
    have tables := tables_of_conditions heap ones teens tens onesTyped teensTyped tensTyped
      onesCondition teensCondition tensCondition
    obtain ⟨extra, optimized⟩ := solve.result n (by omega) heap (context ones teens tens) tables
    exact optimized.returns.supported
  · intro heap n ones teens tens _valid _onesTyped _teensTyped _tensTyped bound _onesCondition _teensCondition _tensCondition
    have bounds : 0 ≤ n ∧ n ≤ 1000 := by
      simpa using (bound_test heap n ones teens tens).condition_iff.mp bound
    lift n to Nat using bounds.1
    obtain ⟨extra, reference⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p17.naive.result
        (globals := context ones teens tens) n heap
    exact reference.returns.supported

theorem requires_solve_0_supported (heap : Mutable.Heap) (n : Int) (ones teens tens : Nat)
    (_valid : heap.Valid)
    (_onesTyped : Mutable.Value.HasType heap (.ref ones) "dict[int,str]")
    (_teensTyped : Mutable.Value.HasType heap (.ref teens) "dict[int,str]")
    (_tensTyped : Mutable.Value.HasType heap (.ref tens) "dict[int,str]") :
    (HeapCalls.evaluate program (conditionEnv n ones teens tens) Solvers.P17Ast.requires_solve_0
      heap (context ones teens tens)).Supported :=
  Execution.supported_of_finishes (bound_test heap n ones teens tens).finishes

theorem requires_solve_1_supported (heap : Mutable.Heap) (n : Int) (ones teens tens : Nat)
    (_valid : heap.Valid)
    (onesTyped : Mutable.Value.HasType heap (.ref ones) "dict[int,str]")
    (_teensTyped : Mutable.Value.HasType heap (.ref teens) "dict[int,str]")
    (_tensTyped : Mutable.Value.HasType heap (.ref tens) "dict[int,str]")
    (_bound : boundCondition heap n ones teens tens)
    : (HeapCalls.evaluate program (fun name => (context ones teens tens).lookup name)
      Solvers.P17Ast.requires_solve_1 heap (context ones teens tens)).Supported := by
  obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType onesTyped rfl
  exact Execution.supported_of_finishes
    (ones_test heap (context ones teens tens) (fun name => (context ones teens tens).lookup name)
      ones entries rfl found).finishes

theorem requires_solve_2_supported (heap : Mutable.Heap) (n : Int) (ones teens tens : Nat)
    (_valid : heap.Valid)
    (_onesTyped : Mutable.Value.HasType heap (.ref ones) "dict[int,str]")
    (teensTyped : Mutable.Value.HasType heap (.ref teens) "dict[int,str]")
    (_tensTyped : Mutable.Value.HasType heap (.ref tens) "dict[int,str]")
    (_bound : boundCondition heap n ones teens tens)
    (_prior1 : tableCondition Solvers.P17Ast.requires_solve_1 heap ones teens tens)
    : (HeapCalls.evaluate program (fun name => (context ones teens tens).lookup name)
      Solvers.P17Ast.requires_solve_2 heap (context ones teens tens)).Supported := by
  obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType teensTyped rfl
  exact Execution.supported_of_finishes
    (teens_test heap (context ones teens tens) (fun name => (context ones teens tens).lookup name)
      teens entries rfl found).finishes

theorem requires_solve_3_supported (heap : Mutable.Heap) (n : Int) (ones teens tens : Nat)
    (_valid : heap.Valid)
    (_onesTyped : Mutable.Value.HasType heap (.ref ones) "dict[int,str]")
    (_teensTyped : Mutable.Value.HasType heap (.ref teens) "dict[int,str]")
    (tensTyped : Mutable.Value.HasType heap (.ref tens) "dict[int,str]")
    (_bound : boundCondition heap n ones teens tens)
    (_prior1 : tableCondition Solvers.P17Ast.requires_solve_1 heap ones teens tens)
    (_prior2 : tableCondition Solvers.P17Ast.requires_solve_2 heap ones teens tens)
    : (HeapCalls.evaluate program (fun name => (context ones teens tens).lookup name)
      Solvers.P17Ast.requires_solve_3 heap (context ones teens tens)).Supported := by
  obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType tensTyped rfl
  exact Execution.supported_of_finishes
    (tens_test heap (context ones teens tens) (fun name => (context ones teens tens).lookup name)
      tens entries rfl found).finishes

end solvers.«17»
