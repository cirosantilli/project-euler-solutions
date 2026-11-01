import solvers.«69.py.lean.d».Trial
import ProjectEulerStatements.Lib.TotientPrimorialScan

namespace solvers.«69»
open FormalSW.PyAst HeapCalls ProjectEulerStatements TotientMaximum
set_option Elab.async false

def working (base : Mutable.Heap) (cache : List Nat) : Mutable.Heap :=
  base ++ [.list (encode cache)]

theorem working_found (base : Mutable.Heap) (cache : List Nat) :
    (working base cache)[base.length]? = some (.list (encode cache)) := by simp [working]

theorem working_store (base : Mutable.Heap) (cache replacement : List Nat) :
    (working base cache).set base.length (.list (encode replacement)) = working base replacement :=
  by simp [working, List.set_append_right]

structure Ready (limit cursor product address : Nat) (env : HeapCalls.Env) : Prop where
  getLimit : env "limit" = some (number limit)
  getCandidate : env "candidate" = some (number cursor)
  getProduct : env "product" = some (number product)
  getCache : env "primes" = some (.ref address)

variable {limit cursor product address : Nat} {env before after : HeapCalls.Env}

theorem Ready.set_flag (ready : Ready limit cursor product address env) (flag : Bool) :
    Ready limit cursor product address (setLocal env "is_prime" (.scalar (.bool flag))) where
  getLimit := by simpa [setLocal] using ready.getLimit
  getCandidate := by simpa [setLocal] using ready.getCandidate
  getProduct := by simpa [setLocal] using ready.getProduct
  getCache := by simpa [setLocal] using ready.getCache

theorem Ready.after_trial (ready : Ready limit cursor product address before)
    (kept : TrialKeeps before after) : Ready limit cursor product address after where
  getLimit := (kept "limit" (by decide) (by decide)).trans ready.getLimit
  getCandidate := (kept "candidate" (by decide) (by decide)).trans ready.getCandidate
  getProduct := (kept "product" (by decide) (by decide)).trans ready.getProduct
  getCache := (kept "primes" (by decide) (by decide)).trans ready.getCache

def nextCache (cache : List Nat) (cursor : Nat) : List Nat :=
  if Nat.Prime cursor then cache ++ [cursor] else cache

abbrev multiplyCode : Stmt := .augAssign (.name "product") .mult (.name "candidate")
abbrev appendCode : Stmt := .expr (.call (.attribute (.name "primes") "append") [.name "candidate"])
abbrev incrementCode : Stmt := .augAssign (.name "candidate") .add (.constant (.int 2))

theorem advance_code_source : scanBody.drop 2 =
    [.ifThen (.name "is_prime") [multiplyCode, appendCode] [], incrementCode] := rfl

theorem scan_code_source : scanBody =
    [.assign [.name "is_prime"] (.constant (.bool true)), trialLoop] ++ scanBody.drop 2 := rfl

theorem advance_result (limit cursor : Nat) (product : Positive) (cache : List Nat)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (ready : Ready limit cursor product.val base.length env)
    (getFlag : env "is_prime" = some (.scalar (.bool (decide (Nat.Prime cursor))))) :
    ∃ finalEnv, BlockContinues program env (scanBody.drop 2) (working base cache)
      finalEnv (working base (nextCache cache cursor)) ∧
      Ready limit (cursor + 2) (nextProduct cursor product).val base.length finalEnv := by
  by_cases prime : Nat.Prime cursor
  · let multiplied := setLocal env "product" (number (product.val * cursor))
    let advanced := setLocal multiplied "candidate" (number (cursor + 2))
    have multiplication : BlockContinues program env [multiplyCode] (working base cache)
        multiplied (working base cache) :=
      BlockContinues.augAssign ready.getProduct (Evaluates.lookup ready.getCandidate)
        (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
    have append : Evaluates program multiplied
        (.call (.attribute (.name "primes") "append") [.name "candidate"])
        (working base cache) (.scalar .none) (working base (cache ++ [cursor])) := by
      have getCache : multiplied "primes" = some (.ref base.length) := by
        simpa [multiplied, setLocal] using ready.getCache
      have getCandidate : multiplied "candidate" = some (number cursor) := by
        simpa [multiplied, setLocal] using ready.getCandidate
      have executed := Evaluates.append (module := program) (globals := []) (env := multiplied)
        (Evaluates.lookup getCache) (working_found base cache)
        (Evaluates.lookup getCandidate) (working_found base cache)
      have encoded : encode cache ++ [number cursor] = encode (cache ++ [cursor]) := by
        simp [encode]
      simpa only [encoded, working_store] using executed
    have appended : BlockContinues program multiplied [appendCode] (working base cache)
        multiplied (working base (cache ++ [cursor])) := BlockContinues.discard append
    have moved : BlockContinues program multiplied [incrementCode]
        (working base (cache ++ [cursor])) advanced (working base (cache ++ [cursor])) :=
      BlockContinues.augAssign
        (by simpa [multiplied, setLocal] using ready.getCandidate) Evaluates.constant
        (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
    refine ⟨advanced, ?_, ?_⟩
    · rw [advance_code_source]
      apply BlockContinues.branch_next (flag := true)
        (by simpa only [prime, decide_true] using Evaluates.lookup getFlag) rfl
      simpa only [nextCache, if_pos prime, List.singleton_append] using
        multiplication.append (appended.append moved)
    · constructor
      · simpa [advanced, multiplied, setLocal] using ready.getLimit
      · rfl
      · simp [advanced, multiplied, setLocal, nextProduct_value, prime]
      · simpa [advanced, multiplied, setLocal] using ready.getCache
  · let advanced := setLocal env "candidate" (number (cursor + 2))
    have moved : BlockContinues program env [incrementCode] (working base cache)
        advanced (working base cache) :=
      BlockContinues.augAssign ready.getCandidate Evaluates.constant
        (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
    refine ⟨advanced, ?_, ?_⟩
    · rw [advance_code_source]
      apply BlockContinues.branch_next (flag := false)
        (by simpa only [prime, decide_false] using Evaluates.lookup getFlag) rfl
      simpa only [nextCache, if_neg prime] using moved
    · constructor
      · simpa [advanced, setLocal] using ready.getLimit
      · rfl
      · simpa [advanced, setLocal, nextProduct_value, prime] using ready.getProduct
      · simpa [advanced, setLocal] using ready.getCache

theorem iteration_result (limit cursor : Nat) (product : Positive) (cache : List Nat)
    (large : 3 ≤ cursor) (ordered : cache.Pairwise (· < ·))
    (complete : PrimeTrialDivision.CompleteBelow cache cursor)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (ready : Ready limit cursor product.val base.length env) :
    ∃ finalEnv, BlockContinues program env scanBody (working base cache)
      finalEnv (working base (nextCache cache cursor)) ∧
      Ready limit (cursor + 2) (nextProduct cursor product).val base.length finalEnv := by
  let flagged := setLocal env "is_prime" (.scalar (.bool true))
  have flagReady : Ready limit cursor product.val base.length flagged := ready.set_flag true
  obtain ⟨testedEnv, tested, flag, kept⟩ := trial_result cursor cache (by omega)
    (ordered.imp (fun less => Nat.le_of_lt less)) complete (working base cache) flagged
    base.length (working_found base cache) flagReady.getCandidate flagReady.getCache rfl
  obtain ⟨finalEnv, advanced, finalReady⟩ :=
    advance_result limit cursor product cache base testedEnv (flagReady.after_trial kept) flag
  refine ⟨finalEnv, ?_, finalReady⟩
  rw [scan_code_source]
  exact (BlockContinues.assign Evaluates.constant).append (tested.append advanced)

end solvers.«69»
