import solvers.«7.py.lean.d».SieveRuntime
import lang.py.FormalSW.HeapRuntimeCostReturns

namespace solvers.«7».scanRuntime
open FormalSW.PyAst HeapCalls.Runtime ProjectEulerSolutions.P7
open markingRuntime
set_option maxRecDepth 4096
set_option maxHeartbeats 1000000
private abbrev program := Solvers.P7Ast.module

def attemptBody : List Stmt :=
  match findFunctionIn program.body "solve" with
  | some (_, statements) => match statements.getLast? with
    | some (.whileDo _ body _) => body
    | _ => []
  | none => []

def scanBody : List Stmt :=
  match (attemptBody[7]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

theorem returns (k i count : Nat) (sieve : Array Bool) :
    ∀ (env : HeapCalls.Env) (heap : Mutable.Heap) (address prime : Nat) (following : List Stmt),
    env "n" = some (.scalar (.int (k + 1 : Nat))) →
    env "count_primes" = some (.scalar (.int count)) →
    env "is_prime" = some (.ref address) →
    heap[address]? = some (.bytearray (sieveBytes sieve)) →
    scanNthPrimeIndex k i count sieve = some prime →
    BlockReturnsIn .word program env
      (.rangeStepLoop "i" i sieve.size 1 scanBody [] :: following) heap (50*(sieve.size-i)+4) (.scalar (.int prime)) heap := by
  induction i, count using scanNthPrimeIndex.induct k sieve with
  | case1 i count finished =>
      intro env heap address prime following target counter buffer found result
      simp [scanNthPrimeIndex.eq_def, finished] at result
  | case2 i ongoing live =>
      intro env heap address prime following target counter buffer found result
      have valid : i < sieve.size := by omega
      have flag : sieve[i] = true := by simpa only [getElem!_pos sieve i valid] using live
      have answer : prime = oddAt i := by simpa [scanNthPrimeIndex.eq_def, ongoing, live] using result.symm
      subst prime
      apply BlockReturnsIn.mono (small := 44) _ (by omega)
      apply BlockReturnsIn.range_step (budget := 40) (by simp [Calls.rangeContinues]; omega)
      apply BlockReturnsIn.branch (cc := 10) (cb := 28) (value := .int 1) (flag := true) (middle := heap)
      · simpa [flag] using sieveRuntime.read
          (HeapCalls.setLocal env "i" (.scalar (.int i))) heap address i sieve valid rfl
          (by simpa [HeapCalls.setLocal] using buffer) found
      · rfl
      · apply BlockReturnsIn.augAssign (ca := 1) (work := 1) (cb := 21) (left := .int k)
          (right := .int 1) (value := .scalar (.int (k + 1 : Nat)))
          (middle := heap)
        · simpa [HeapCalls.setLocal] using counter
        · exact EvaluatesIn.constant
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
        · rfl
        · apply BlockReturnsIn.branch (cc := 7) (cb := 12) (value := .bool true) (flag := true) (middle := heap)
          · exact EvaluatesIn.compare (work := 2) (av := .scalar (.int (k + 1 : Nat)))
              (bv := .scalar (.int (k + 1 : Nat))) (EvaluatesIn.lookup rfl)
              (EvaluatesIn.lookup (by simpa [HeapCalls.setLocal] using target))
              (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]) rfl
          · rfl
          · apply BlockReturnsIn.return_value (cost := 11)
            exact EvaluatesIn.binary (av := .int (2 * i : Nat)) (bv := .int 1) (work := 1)
              (middle := heap) (afterRight := heap)
              (EvaluatesIn.binary (av := .int 2) (bv := .int i) (work := 1)
                EvaluatesIn.constant (EvaluatesIn.lookup (by simp [HeapCalls.setLocal]))
                (by simp [Mutable.binary, scalarBinary]) rfl)
              EvaluatesIn.constant (by simp [Mutable.binary, scalarBinary, oddAt]) rfl
  | case3 i count ongoing live mismatch ih =>
      intro env heap address prime following target counter buffer found result
      have valid : i < sieve.size := by omega
      have flag : sieve[i] = true := by simpa only [getElem!_pos sieve i valid] using live
      have resultNext : scanNthPrimeIndex k (i + 1) (count + 1) sieve = some prime := by
        rwa [scanNthPrimeIndex.eq_def, dif_neg ongoing, if_pos live, if_neg mismatch] at result
      apply BlockReturnsIn.mono (small := (10 + (1 + 1 + 5 + (7 + (50*(sieve.size-(i+1))+4) + 2)) + 2) + 4) _ (by omega)
      apply BlockReturnsIn.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockReturnsIn.branch (cc := 10) (value := .int 1) (flag := true) (middle := heap)
      · simpa [flag] using sieveRuntime.read
          (HeapCalls.setLocal env "i" (.scalar (.int i))) heap address i sieve valid rfl
          (by simpa [HeapCalls.setLocal] using buffer) found
      · rfl
      · apply BlockReturnsIn.augAssign (ca := 1) (work := 1) (left := .int count)
          (right := .int 1) (value := .scalar (.int (count + 1 : Nat)))
          (middle := heap)
        · simpa [HeapCalls.setLocal] using counter
        · exact EvaluatesIn.constant
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
        · rfl
        · apply BlockReturnsIn.branch (cc := 7) (value := .bool false) (flag := false) (middle := heap)
          · exact EvaluatesIn.compare (work := 2) (av := .scalar (.int (count + 1 : Nat)))
              (bv := .scalar (.int (k + 1 : Nat))) (EvaluatesIn.lookup rfl)
              (EvaluatesIn.lookup (by simpa [HeapCalls.setLocal] using target))
              (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, mismatch]) rfl
          · rfl
          · simpa only [Int.natCast_add, Int.natCast_one] using
              ih _ heap address prime following (by simpa [HeapCalls.setLocal] using target) rfl
                (by simpa [HeapCalls.setLocal] using buffer) found resultNext
  | case4 i count ongoing dead ih =>
      intro env heap address prime following target counter buffer found result
      have valid : i < sieve.size := by omega
      have flag : sieve[i] = false := by
        have negative : sieve[i] ≠ true := by simpa only [getElem!_pos sieve i valid] using dead
        exact Bool.eq_false_iff.mpr negative
      have resultNext : scanNthPrimeIndex k (i + 1) count sieve = some prime := by
        rwa [scanNthPrimeIndex.eq_def, dif_neg ongoing, if_neg dead] at result
      apply BlockReturnsIn.mono (small := (10 + (50*(sieve.size-(i+1))+4) + 2) + 4) _ (by omega)
      apply BlockReturnsIn.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockReturnsIn.branch (cc := 10) (value := .int 0) (flag := false) (middle := heap)
      · simpa [flag] using sieveRuntime.read
          (HeapCalls.setLocal env "i" (.scalar (.int i))) heap address i sieve valid rfl
          (by simpa [HeapCalls.setLocal] using buffer) found
      · rfl
      · simpa only [Int.natCast_add, Int.natCast_one] using
          ih _ heap address prime following (by simpa [HeapCalls.setLocal] using target)
            (by simpa [HeapCalls.setLocal] using counter)
            (by simpa [HeapCalls.setLocal] using buffer) found resultNext


private def scanHitBody : List Stmt :=
  match (scanBody[0]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []

private theorem Stable.counter (env : HeapCalls.Env) (count : Nat) :
    Stable env (HeapCalls.setLocal env "count_primes" (.scalar (.int count))) := by
  intro name member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl <;> rfl

/-- An exhausted source scan preserves the inputs needed to retry with a larger limit. -/
theorem continues (k i count : Nat) (sieve : Array Bool) :
    ∀ (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat),
    env "n" = some (.scalar (.int (k + 1 : Nat))) →
    env "count_primes" = some (.scalar (.int count)) →
    env "is_prime" = some (.ref address) →
    heap[address]? = some (.bytearray (sieveBytes sieve)) →
    scanNthPrimeIndex k i count sieve = none →
    ∃ finalEnv,
      BlockContinuesIn .word program env [.rangeStepLoop "i" i sieve.size 1 scanBody []]
        heap (50*(sieve.size-i)+4) finalEnv heap ∧ Stable env finalEnv := by
  induction i, count using scanNthPrimeIndex.induct k sieve with
  | case1 i count finished =>
      intro env heap address target counter buffer found result
      refine ⟨env, ?_, Stable.refl env⟩
      exact (BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega)
        BlockContinuesIn.nil).mono (by omega)
  | case2 i ongoing live =>
      intro env heap address target counter buffer found result
      simp [scanNthPrimeIndex.eq_def, ongoing, live] at result
  | case3 i count ongoing live mismatch ih =>
      intro env heap address target counter buffer found result
      have valid : i < sieve.size := by omega
      have flag : sieve[i] = true := by simpa only [getElem!_pos sieve i valid] using live
      have resultNext : scanNthPrimeIndex k (i + 1) (count + 1) sieve = none := by
        rwa [scanNthPrimeIndex.eq_def, dif_neg ongoing, if_pos live, if_neg mismatch] at result
      let indexed := HeapCalls.setLocal env "i" (.scalar (.int i))
      let advanced := HeapCalls.setLocal indexed "count_primes" (.scalar (.int (count + 1 : Nat)))
      have stable : Stable env advanced := (Stable.index env i).trans
        (Stable.counter indexed (count + 1))
      obtain ⟨finalEnv, rest, preserved⟩ := ih advanced heap address
        ((stable "n" (by simp)).trans target) rfl ((stable "is_prime" (by simp)).trans buffer) found resultNext
      refine ⟨finalEnv, ?_, stable.trans preserved⟩

      have advance : BlockContinuesIn .word program indexed (scanHitBody.take 1) heap 7 advanced heap := by
        exact BlockContinuesIn.augAssign (left := .int count) (right := .int 1) (work := 1)
          (by simpa [indexed, HeapCalls.setLocal] using counter) EvaluatesIn.constant
          (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary]) rfl
      have skip : BlockContinuesIn .word program advanced (scanHitBody.drop 1) heap 9 advanced heap := by
        apply BlockContinuesIn.branch (cc := 7) (cb := 0) (value := .bool false) (flag := false) (middle := heap)
        · exact EvaluatesIn.compare (work := 2) (av := .scalar (.int (count + 1 : Nat)))
            (bv := .scalar (.int (k + 1 : Nat))) (EvaluatesIn.lookup rfl)
            (EvaluatesIn.lookup ((stable "n" (by simp)).trans target))
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, mismatch]) rfl
        · rfl
        · exact BlockContinuesIn.nil
      have iteration : BlockContinuesIn .word program indexed scanBody heap 28 advanced heap := by
        apply BlockContinuesIn.branch (cc := 10) (cb := 16) (value := .int 1) (flag := true) (middle := heap)
        · simpa [flag] using sieveRuntime.read indexed heap address i sieve valid rfl
            (by simpa [indexed, HeapCalls.setLocal] using buffer) found
        · rfl
        · exact advance.append skip
      have ranged := BlockContinuesIn.range_step (env := env) (name := "i") (next := (i : Int))
        (stop := (sieve.size : Int)) (stride := 1) (body := scanBody)
        (by simp [Calls.rangeContinues]; omega)
        (by simpa only [Int.natCast_add, Int.natCast_one] using iteration.append rest)
      exact ranged.mono (by omega)
  | case4 i count ongoing dead ih =>
      intro env heap address target counter buffer found result
      have valid : i < sieve.size := by omega
      have flag : sieve[i] = false := by
        have negative : sieve[i] ≠ true := by simpa only [getElem!_pos sieve i valid] using dead
        exact Bool.eq_false_iff.mpr negative
      have resultNext : scanNthPrimeIndex k (i + 1) count sieve = none := by
        rwa [scanNthPrimeIndex.eq_def, dif_neg ongoing, if_neg dead] at result
      let indexed := HeapCalls.setLocal env "i" (.scalar (.int i))
      have stable := Stable.index env i
      obtain ⟨finalEnv, rest, preserved⟩ := ih indexed heap address
        ((stable "n" (by simp)).trans target) (by simpa [indexed, HeapCalls.setLocal] using counter)
        ((stable "is_prime" (by simp)).trans buffer) found resultNext
      refine ⟨finalEnv, ?_, stable.trans preserved⟩

      have iteration : BlockContinuesIn .word program indexed scanBody heap 12 indexed heap := by
        apply BlockContinuesIn.branch (cc := 10) (cb := 0) (value := .int 0) (flag := false) (middle := heap)
        · simpa [flag] using sieveRuntime.read indexed heap address i sieve valid rfl
            (by simpa [indexed, HeapCalls.setLocal] using buffer) found
        · rfl
        · exact BlockContinuesIn.nil
      have ranged := BlockContinuesIn.range_step (env := env) (name := "i") (next := (i : Int))
        (stop := (sieve.size : Int)) (stride := 1) (body := scanBody)
        (by simp [Calls.rangeContinues]; omega)
        (by simpa only [Int.natCast_add, Int.natCast_one] using iteration.append rest)
      exact ranged.mono (by omega)

/-- Counter initialization and the successful range scan. -/
theorem source_returns (k prime : Nat) (sieve : Array Bool) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (following : List Stmt)
    (target : env "n" = some (.scalar (.int (k+1 : Nat))))
    (size : env "size" = some (.scalar (.int sieve.size)))
    (buffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve)))
    (result : scanNthPrimeIndex k 1 1 sieve = some prime) :
    BlockReturnsIn .word program env ((attemptBody.drop 6).take 2 ++ following) heap
      (50*sieve.size+16) (.scalar (.int prime)) heap := by
  let initialized := HeapCalls.setLocal env "count_primes" (.scalar (.int 1))
  have scanning := returns k 1 1 sieve initialized heap address prime following
    (by simpa [initialized, HeapCalls.setLocal] using target) rfl
    (by simpa [initialized, HeapCalls.setLocal] using buffer) found result
  have ranged := BlockReturnsIn.for_range_pair (a := .constant (.int 1)) (b := .name "size")
    EvaluatesIn.constant (EvaluatesIn.lookup (by simpa [initialized, HeapCalls.setLocal] using size)) scanning
  have assigned := BlockReturnsIn.assign (name := "count_primes")
    (env := env) EvaluatesIn.constant ranged
  exact assigned.mono (by omega)

/-- An exhausted scan preserves the input and the sieve for the retry. -/
theorem source_continues (k : Nat) (sieve : Array Bool) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat)
    (target : env "n" = some (.scalar (.int (k+1 : Nat))))
    (size : env "size" = some (.scalar (.int sieve.size)))
    (buffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve)))
    (result : scanNthPrimeIndex k 1 1 sieve = none) :
    ∃ finalEnv, BlockContinuesIn .word program env ((attemptBody.drop 6).take 2) heap
      (50*sieve.size+16) finalEnv heap ∧ Stable env finalEnv := by
  let initialized := HeapCalls.setLocal env "count_primes" (.scalar (.int 1))
  have stable := Stable.counter env 1
  obtain ⟨finalEnv, scanning, preserved⟩ := continues k 1 1 sieve initialized heap address
    ((stable "n" (by simp)).trans target) rfl ((stable "is_prime" (by simp)).trans buffer) found result
  have ranged := BlockContinuesIn.for_range_pair (a := .constant (.int 1)) (b := .name "size")
    EvaluatesIn.constant (EvaluatesIn.lookup ((stable "size" (by simp)).trans size)) scanning
  have assigned := (BlockContinuesIn.assign (name := "count_primes") (env := env)
    (module := program) (heap := heap) (arithmetic := .word) (value := .scalar (.int 1)) EvaluatesIn.constant).append ranged
  exact ⟨finalEnv, assigned.mono (by omega), stable.trans preserved⟩

end solvers.«7».scanRuntime
