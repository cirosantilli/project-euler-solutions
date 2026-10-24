import «.formalsw».solvers.«69.py.ast»
import lang.py.FormalSW.LoopCalls
import ProjectEulerStatements.Lib.PrimeTrialDivision

namespace solvers.«69»
open FormalSW.PyAst HeapCalls ProjectEulerStatements
set_option Elab.async false

abbrev program := Solvers.P69Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
def encode (values : List Nat) : List Mutable.Value := values.map number
abbrev solverArgs : List Arg := ((findFunctionIn program.body "solve").getD ([], [])).1
abbrev solverBody : List Stmt := ((findFunctionIn program.body "solve").getD ([], [])).2
abbrev solverCode : List Stmt := executableBody solverBody
abbrev scanBody : List Stmt :=
  match (solverCode[4]? : Option Stmt) with
  | some (.whileDo _ instructions _) => instructions
  | _ => []
abbrev trialBody : List Stmt :=
  match (scanBody[1]? : Option Stmt) with
  | some (.forIn _ _ instructions _) => instructions
  | _ => []
abbrev squareCondition : Expr :=
  .compare (.binOp (.name "divisor") .mult (.name "divisor")) [(.gt, .name "candidate")]
abbrev divisorCondition : Expr :=
  .compare (.binOp (.name "candidate") .mod (.name "divisor")) [(.eq, .constant (.int 0))]
abbrev trialLoop : Stmt := .forIn (.name "divisor") (.name "primes") trialBody []

def TrialKeeps (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ≠ "divisor" → name ≠ "is_prime" → after name = before name

theorem trial_code_source : trialBody =
    [.ifThen squareCondition [.breakLoop] [],
     .ifThen divisorCondition
       [.assign [.name "is_prime"] (.constant (.bool false)), .breakLoop] []] := rfl

theorem trial_loop_source : scanBody[1]? = some trialLoop := rfl

theorem square_condition_result (n p : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "candidate" = some (number n)) (getP : env "divisor" = some (number p)) :
    Evaluates program env squareCondition heap (.scalar (.bool (decide (n < p * p)))) heap := by
  apply Evaluates.compare (av := number (p * p)) (bv := number n)
    (middle := heap) (afterRight := heap)
  · exact Evaluates.binary (middle := heap) (afterRight := heap)
      (Evaluates.lookup getP) (Evaluates.lookup getP)
      (by simp [Mutable.binary, scalarBinary])
  · exact Evaluates.lookup getN
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    norm_cast

theorem divisor_condition_result (n p : Nat) (prime : Nat.Prime p)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "candidate" = some (number n)) (getP : env "divisor" = some (number p)) :
    Evaluates program env divisorCondition heap (.scalar (.bool (decide (n % p = 0)))) heap := by
  apply Evaluates.compare (av := number (n % p)) (bv := number 0)
    (middle := heap) (afterRight := heap)
  · exact Evaluates.binary (middle := heap) (afterRight := heap)
      (Evaluates.lookup getN) (Evaluates.lookup getP)
      (by simp [Mutable.binary, scalarBinary, prime.pos, prime.ne_zero])
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    norm_cast

theorem trial_loop_result (n : Nat) (cache remaining : List Nat)
    (primes : ∀ p ∈ cache, Nat.Prime p) (position : Nat) (suffix : cache.drop position = remaining)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat)
    (found : heap[address]? = some (.list (encode cache)))
    (getN : env "candidate" = some (number n))
    (getFlag : env "is_prime" = some (.scalar (.bool true))) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "divisor" (sequenceSource (.ref address)) position trialBody []]
      heap finalEnv heap ∧
      finalEnv "is_prime" = some (.scalar (.bool (PrimeTrialDivision.trial n remaining))) ∧
      TrialKeeps env finalEnv := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (encode cache), heap) := by
    simp [Mutable.sequence, found]
  induction remaining generalizing position env with
  | nil =>
      have missing : cache[position]? = none := by
        simpa only [suffix, List.head?_nil] using
          (List.head?_drop (l := cache) (i := position)).symm
      refine ⟨env, ?_, getFlag, fun _ _ _ => rfl⟩
      apply BlockContinues.sequence_done (source := sequenceSource (.ref address))
        (items := encode cache) sequence
        (by simp [encode, List.getElem?_map, missing]) .nil
  | cons p rest ih =>
      have item : cache[position]? = some p := by
        simpa only [suffix, List.head?_cons] using
          (List.head?_drop (l := cache) (i := position)).symm
      obtain ⟨within, equal⟩ := List.getElem?_eq_some_iff.mp item
      have prime : Nat.Prime p := primes p (equal ▸ List.getElem_mem within)
      let indexed := setLocal env "divisor" (number p)
      have nIndexed : indexed "candidate" = some (number n) := by
        simpa [indexed, setLocal] using getN
      have flagIndexed : indexed "is_prime" = some (.scalar (.bool true)) := by
        simpa [indexed, setLocal] using getFlag
      have squareTest := square_condition_result n p heap indexed nIndexed rfl
      by_cases stop : n < p * p
      · refine ⟨indexed, ?_, ?_, ?_⟩
        · apply BlockContinues.sequence_step (source := sequenceSource (.ref address))
            (items := encode cache) (value := number p) sequence (by simp [encode, item])
          rw [trial_code_source]
          apply BlockContinues.branch_next (flag := true)
            (by simpa only [stop, decide_true] using squareTest) rfl
          exact BlockContinues.break_loop rfl .nil
        · simpa only [PrimeTrialDivision.trial, Nat.sqrt_lt, stop, if_true] using flagIndexed
        · intro name notDivisor notFlag
          simp [indexed, setLocal, notDivisor]
      · have divisorTest := divisor_condition_result n p prime heap indexed nIndexed rfl
        by_cases divides : n % p = 0
        · let flagged := setLocal indexed "is_prime" (.scalar (.bool false))
          refine ⟨flagged, ?_, ?_, ?_⟩
          · apply BlockContinues.sequence_step (source := sequenceSource (.ref address))
              (items := encode cache) (value := number p) sequence (by simp [encode, item])
            rw [trial_code_source]
            apply BlockContinues.branch_next (flag := false)
              (by simpa only [stop, decide_false] using squareTest) rfl
            apply BlockContinues.branch_next (flag := true)
              (by simpa only [divides, decide_true] using divisorTest) rfl
            apply (BlockContinues.assign Evaluates.constant).append
            exact BlockContinues.break_loop rfl .nil
          · simp [PrimeTrialDivision.trial, Nat.sqrt_lt, stop, divides, flagged, setLocal]
          · intro name notDivisor notFlag
            simp [flagged, indexed, setLocal, notDivisor, notFlag]
        · obtain ⟨finalEnv, scanned, finalFlag, kept⟩ := ih (position + 1)
            (by simp [List.drop_add_one_eq_tail_drop, suffix]) indexed nIndexed flagIndexed
          refine ⟨finalEnv, ?_, ?_, ?_⟩
          · apply BlockContinues.sequence_step (source := sequenceSource (.ref address))
              (items := encode cache) (value := number p) sequence (by simp [encode, item])
            rw [trial_code_source]
            apply BlockContinues.branch_next (flag := false)
              (by simpa only [stop, decide_false] using squareTest) rfl
            apply BlockContinues.branch_next (flag := false)
              (by simpa only [divides, decide_false] using divisorTest) rfl
            exact BlockContinues.loop_resume scanned
          · simpa only [PrimeTrialDivision.trial, Nat.sqrt_lt, stop, if_false, divides]
              using finalFlag
          · intro name notDivisor notFlag
            rw [kept name notDivisor notFlag]
            simp [indexed, setLocal, notDivisor]

theorem trial_result (n : Nat) (cache : List Nat) (large : 2 ≤ n)
    (ordered : cache.Pairwise (· ≤ ·)) (complete : PrimeTrialDivision.CompleteBelow cache n)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat)
    (found : heap[address]? = some (.list (encode cache)))
    (getN : env "candidate" = some (number n)) (getCache : env "primes" = some (.ref address))
    (getFlag : env "is_prime" = some (.scalar (.bool true))) :
    ∃ finalEnv, BlockContinues program env [trialLoop] heap finalEnv heap ∧
      finalEnv "is_prime" = some (.scalar (.bool (decide (Nat.Prime n)))) ∧
      TrialKeeps env finalEnv := by
  obtain ⟨finalEnv, scanned, flag, kept⟩ := trial_loop_result n cache cache
    (fun p member => ((complete p).mp member).1) 0 rfl heap env address found getN getFlag
  refine ⟨finalEnv, ?_, ?_, kept⟩
  · exact BlockContinues.for_sequence (items := encode cache)
      (by intros expressions impossible; cases impossible)
      (Evaluates.lookup getCache) (by simp [Mutable.sequence, found]) scanned
  · simpa only [PrimeTrialDivision.trial_complete n cache large ordered complete] using flag

end solvers.«69»
