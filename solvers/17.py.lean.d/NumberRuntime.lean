import solvers.«17.py.lean.d».BelowHundredRuntime

namespace solvers.«17».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open ProjectEulerStatements.P17
private abbrev program := Solvers.P17Ast.module
private abbrev numberBody : List Stmt :=
  executableBody ((findFunctionIn program.body "letters_in_number").getD ([], [])).2

theorem hundreds_cost (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tables : Tables heap globals) (env : HeapCalls.Env) (n : Nat)
    (lo : 100 ≤ n) (hi : n < 1000)
    (getN : env "n" = some (.scalar (.int n)))
    (getOnes : env "ONES" = globals.lookup "ONES") :
    ∃ extra, BlockReturnsIn .word program env (numberBody.drop 3) heap 500
      (.scalar (.int (wordLen n))) (heap ++ extra) globals := by
  let pair : Mutable.Object := .tuple
    [.scalar (.int ((n / 100 : Nat) : Int)), .scalar (.int ((n % 100 : Nat) : Int))]
  let unpacked := assignNames env ["h", "r"]
    [.scalar (.int ((n / 100 : Nat) : Int)), .scalar (.int ((n % 100 : Nat) : Int))]
  let total := smallWordLen (n / 100) + 7
  let ready := setLocal unpacked "total" (.scalar (.int total))
  have getR : ready "r" = some (.scalar (.int ((n % 100 : Nat) : Int))) := by
    simp [ready, unpacked, assignNames, setLocal]
  have finish : ∃ extra, BlockReturnsIn .word program ready (numberBody.drop 5)
      (heap ++ [pair]) 300 (.scalar (.int (wordLen n))) ((heap ++ [pair]) ++ extra) globals := by
    rw [wordLen_hundreds n lo hi]
    by_cases zero : n % 100 = 0
    · refine ⟨[], ?_⟩
      simp only [if_pos zero, Nat.add_zero, List.append_nil]
      apply BlockReturnsIn.mono
      · apply BlockReturnsIn.branch (flag := false)
          (EvaluatesIn.compare (work := 2) (value := .scalar (.bool false))
            (.lookup getR) .constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero]) rfl) rfl
        exact BlockReturnsIn.return_value (.lookup (by simp [ready, setLocal, total]))
      · decide
    · obtain ⟨extra, called⟩ := below_hundred_cost (heap ++ [pair]) globals (tables.append [pair])
        (n % 100) (by omega) (by omega)
      refine ⟨extra, ?_⟩
      rw [if_neg zero]
      apply BlockReturnsIn.mono
      · apply BlockReturnsIn.branch (flag := true)
          (EvaluatesIn.compare (work := 2) (value := .scalar (.bool true))
            (.lookup getR) .constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; exact_mod_cast zero) rfl) rfl
        apply BlockReturnsIn.augAssign (left := .int total) (right := .int 3)
          (value := .scalar (.int (total + 3))) (work := 1)
          (middle := heap ++ [pair]) (by simp [ready, setLocal])
        · exact length_constant _ _ globals "and"
        · rfl
        · rfl
        · apply BlockReturnsIn.branch (flag := true)
            (EvaluatesIn.compare (work := 2) (value := .scalar (.bool true))
              (.lookup (by simpa [setLocal] using getR)) .constant
              (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega) rfl) rfl
          apply BlockReturnsIn.augAssign (left := .int (total + 3))
            (right := .int (wordLen (n % 100))) (work := 1)
            (value := .scalar (.int ((total + 3 : Nat) + wordLen (n % 100))))
            (middle := (heap ++ [pair]) ++ extra) (by simp [setLocal])
          · exact EvaluatesIn.call_single (.lookup (by simpa [setLocal] using getR)) called
          · simp [inplace, Mutable.binary, scalarBinary]
          · rfl
          · apply BlockReturnsIn.return_value
            apply EvaluatesIn.lookup
            simp [setLocal, total]
            omega
      · decide
  obtain ⟨extra, finish⟩ := finish
  refine ⟨[pair] ++ extra, ?_⟩
  have divided := EvaluatesIn.divmod_ints (globals := globals) (module := program)
    (env := env) (heap := heap) (EvaluatesIn.lookup getN)
    (EvaluatesIn.constant (value := .int 100)) (by decide)
  have quotient : (n : Int).fdiv 100 = ((n / 100 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide)]; rfl
  have remaining : (n : Int) - 100 * (n : Int).fdiv 100 = ((n % 100 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide), ← Int.emod_def]; rfl
  rw [remaining, quotient] at divided
  apply BlockReturnsIn.mono
  · apply BlockContinuesIn.returns (BlockContinuesIn.unpack_tuple
      (targets := [.name "h", .name "r"])
          (items := [.scalar (.int ((n / 100 : Nat) : Int)), .scalar (.int ((n % 100 : Nat) : Int))])
          rfl divided (by simp) rfl)
    apply BlockReturnsIn.assign (value := .scalar (.int total)) (middle := heap ++ [pair])
    · apply EvaluatesIn.binary (work := 1)
      · exact table_read _ globals (tables.1.append [pair]) _ "h" (n / 100)
          (by simpa [assignNames, setLocal] using getOnes)
          (by simp [assignNames, setLocal]) (by omega) (by omega)
      · exact length_constant _ _ globals "hundred"
      · simp [total, Mutable.binary, scalarBinary]
        decide
      · rfl
    · simpa only [ready, unpacked, List.append_assoc] using finish
  · change 375 ≤ 500
    decide

theorem number_block (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tables : Tables heap globals) (env : HeapCalls.Env) (n : Nat)
    (lo : 1 ≤ n) (hi : n ≤ 1000)
    (getN : env "n" = some (.scalar (.int n)))
    (getOnes : env "ONES" = globals.lookup "ONES") :
    ∃ extra, BlockReturnsIn .word program env numberBody heap 600
      (.scalar (.int (wordLen n))) (heap ++ extra) globals := by
  have guard := bounded_cost env heap globals "n" n 1 1000 getN (by omega)
  simp only [show (n : Int) ≤ 1000 by omega, decide_true] at guard
  by_cases thousand : n = 1000
  · subst n
    refine ⟨[], ?_⟩
    simp only [List.append_nil, wordLen_thousand]
    apply BlockReturnsIn.mono
    · apply BlockReturnsIn.assertion guard
      apply BlockReturnsIn.branch (flag := true)
        (EvaluatesIn.compare (work := 2) (value := .scalar (.bool true))
          (.lookup getN) .constant rfl rfl) rfl
      apply BlockReturnsIn.return_value
      exact EvaluatesIn.binary (work := 1)
        (length_constant _ _ globals "one") (length_constant _ _ globals "thousand") rfl rfl
    · decide
  · have thousandGuard := EvaluatesIn.compare (globals := globals) (arithmetic := .word)
      (module := program) (env := env) (heap := heap) (op := .eq) (work := 2)
      (value := .scalar (.bool false)) (EvaluatesIn.lookup getN)
      (EvaluatesIn.constant (value := .int 1000))
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, show (n : Int) ≠ 1000 by omega]) rfl
    have smallGuard := EvaluatesIn.compare (globals := globals) (arithmetic := .word)
      (module := program) (env := env) (heap := heap) (op := .lt) (work := 2)
      (value := .scalar (.bool (decide (n < 100)))) (EvaluatesIn.lookup getN)
      (EvaluatesIn.constant (value := .int 100))
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]) rfl
    by_cases small : n < 100
    · obtain ⟨extra, called⟩ := below_hundred_cost heap globals tables n lo (by omega)
      refine ⟨extra, ?_⟩
      apply BlockReturnsIn.mono
      · apply BlockReturnsIn.assertion guard
        apply BlockReturnsIn.branch (flag := false) thousandGuard rfl
        apply BlockReturnsIn.branch (flag := true) (by simpa [small] using smallGuard) rfl
        exact BlockReturnsIn.return_value (EvaluatesIn.call_single (.lookup getN) called)
      · decide
    · obtain ⟨extra, tail⟩ := hundreds_cost heap globals tables env n
        (by omega) (by omega) getN getOnes
      refine ⟨extra, ?_⟩
      apply BlockReturnsIn.mono
      · apply BlockReturnsIn.assertion guard
        apply BlockReturnsIn.branch (flag := false) thousandGuard rfl
        apply BlockReturnsIn.branch (flag := false) (by simpa [small] using smallGuard) rfl
        exact tail
      · decide

theorem number_cost (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tables : Tables heap globals) (n : Nat) (lo : 1 ≤ n) (hi : n ≤ 1000) :
    ∃ extra, ReturnsIn .word program "letters_in_number" [.scalar (.int n)] heap 603
      (.scalar (.int (wordLen n))) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "letters_in_number").getD ([], [])).1
  let body := ((findFunctionIn program.body "letters_in_number").getD ([], [])).2
  obtain ⟨extra, executed⟩ := number_block heap globals tables
    (functionEnv args body [.scalar (.int n)] globals) n lo hi
    (by cases globals <;> rfl)
    (by cases globals with
        | nil => rfl
        | cons entry rest =>
            change (if "ONES" ∈ blockLocals body then none else _) = _
            rw [if_neg (by
              simp [body, program, Solvers.P17Ast.module, findFunctionIn,
                blockLocals, assignedLocals, targetLocals])])
  exact ⟨extra, BlockReturnsIn.function rfl rfl executed⟩

end solvers.«17».runtimeCost
