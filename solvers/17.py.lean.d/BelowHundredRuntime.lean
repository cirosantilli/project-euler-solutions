import solvers.«17.py.lean.d».TableRuntime

namespace solvers.«17».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open ProjectEulerStatements.P17
private abbrev program := Solvers.P17Ast.module
private abbrev belowHundredBody : List Stmt :=
  executableBody ((findFunctionIn program.body "letters_1_to_99").getD ([], [])).2

theorem below_hundred_block (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tables : Tables heap globals) (env : HeapCalls.Env) (n : Nat)
    (lo : 1 ≤ n) (hi : n ≤ 99)
    (getN : env "n" = some (.scalar (.int n)))
    (getOnes : env "ONES" = globals.lookup "ONES")
    (getTeens : env "TEENS" = globals.lookup "TEENS")
    (getTens : env "TENS" = globals.lookup "TENS") :
    ∃ extra, BlockReturnsIn .word program env belowHundredBody heap 200
      (.scalar (.int (wordLen n))) (heap ++ extra) globals := by
  have guard := bounded_cost env heap globals "n" n 1 99 getN (by omega)
  simp only [show (n : Int) ≤ 99 by omega, decide_true] at guard
  have onesGuard := EvaluatesIn.compare (globals := globals) (arithmetic := .word)
    (module := program) (env := env) (heap := heap) (op := .lt) (work := 2)
    (value := .scalar (.bool (decide (n < 10))))
    (EvaluatesIn.lookup getN) (EvaluatesIn.constant (value := .int 10))
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]) rfl
  by_cases ones : n < 10
  · refine ⟨[], ?_⟩
    simp only [List.append_nil]
    apply BlockReturnsIn.mono
    · apply BlockReturnsIn.assertion guard
      apply BlockReturnsIn.branch (flag := true) (by simpa [ones] using onesGuard) rfl
      rw [wordLen_below_twenty n (by omega)]
      exact BlockReturnsIn.return_value
        (table_read heap globals tables.1 env "n" n getOnes getN (by omega) ones)
    · decide
  · have teenGuard := bounded_cost env heap globals "n" n 10 19 getN (by omega)
    by_cases teens : n < 20
    · refine ⟨[], ?_⟩
      simp only [List.append_nil]
      apply BlockReturnsIn.mono
      · apply BlockReturnsIn.assertion guard
        apply BlockReturnsIn.branch (flag := false) (by simpa [ones] using onesGuard) rfl
        apply BlockReturnsIn.branch (flag := true)
          (by simpa only [show (n : Int) ≤ 19 by omega, decide_true] using teenGuard) rfl
        rw [wordLen_below_twenty n teens]
        exact BlockReturnsIn.return_value
          (table_read heap globals tables.2.1 env "n" n getTeens getN (by omega) teens)
      · decide
    · let pair : Mutable.Object := .tuple
        [.scalar (.int ((n / 10 : Nat) : Int)), .scalar (.int ((n % 10 : Nat) : Int))]
      refine ⟨[pair], ?_⟩
      apply BlockReturnsIn.mono
      · apply BlockReturnsIn.assertion guard
        apply BlockReturnsIn.branch (flag := false) (by simpa [ones] using onesGuard) rfl
        apply BlockReturnsIn.branch (flag := false)
          (by simpa only [show ¬ (n : Int) ≤ 19 by omega, decide_false] using teenGuard) rfl
        have divided := EvaluatesIn.divmod_ints (globals := globals) (module := program)
          (env := env) (heap := heap) (EvaluatesIn.lookup getN)
          (EvaluatesIn.constant (value := .int 10)) (by decide)
        have quotient : (n : Int).fdiv 10 = ((n / 10 : Nat) : Int) := by
          rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide)]; rfl
        have remaining : (n : Int) - 10 * (n : Int).fdiv 10 = ((n % 10 : Nat) : Int) := by
          rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide), ← Int.emod_def]; rfl
        rw [remaining, quotient] at divided
        apply BlockContinuesIn.returns (BlockContinuesIn.unpack_tuple
          (targets := [.name "t", .name "u"])
          (items := [.scalar (.int ((n / 10 : Nat) : Int)), .scalar (.int ((n % 10 : Nat) : Int))])
          rfl divided (by simp) rfl)
        rw [wordLen_below_hundred n (by omega), if_neg teens]
        apply BlockReturnsIn.return_value
        apply EvaluatesIn.binary (work := 1)
        · exact table_read _ globals (tables.2.2.append [pair]) _ "t" (n / 10)
            (by simpa [assignNames, setLocal] using getTens)
            (by simp [assignNames, setLocal]) (by omega) (by omega)
        · exact table_read _ globals (tables.1.append [pair]) _ "u" (n % 10)
            (by simpa [assignNames, setLocal] using getOnes)
            (by simp [assignNames, setLocal]) (by omega) (by omega)
        · simp [Mutable.binary, scalarBinary]
        · rfl
      · change 152 ≤ 200
        decide

theorem below_hundred_cost (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tables : Tables heap globals) (n : Nat) (lo : 1 ≤ n) (hi : n ≤ 99) :
    ∃ extra, ReturnsIn .word program "letters_1_to_99" [.scalar (.int n)] heap 203
      (.scalar (.int (wordLen n))) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "letters_1_to_99").getD ([], [])).1
  let body := ((findFunctionIn program.body "letters_1_to_99").getD ([], [])).2
  obtain ⟨extra, executed⟩ := below_hundred_block heap globals tables
    (functionEnv args body [.scalar (.int n)] globals) n lo hi
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
  exact ⟨extra, BlockReturnsIn.function rfl rfl executed⟩

end solvers.«17».runtimeCost
