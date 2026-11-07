import solvers.«7.py.lean.d».RetryRuntime
import solvers.«7.py.lean.d».InitialRuntime

namespace solvers.«7».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime ProjectEulerSolutions.P7
open scanRuntime (attemptBody)
private abbrev program := Solvers.P7Ast.module

/-- A failed attempt starts below the target prime; doubling overshoots by at
most a factor of two. Retain the initial estimate when it is larger. -/
noncomputable def searchLimit (n : Nat) : Nat :=
  max (initialLimit n) (2*ProjectEulerStatements.P7.naive n)

theorem source_bound (k : Nat) :
    TimeBound .word program "solve" [.scalar (.int (k+2 : Nat))] []
      (2000*searchLimit (k+2)*(Nat.log2 (searchLimit (k+2))+1)+100) := by
  let env := argumentEnv [⟨"n", some (.name "int"), none⟩] [.scalar (.int (k+2 : Nat))]
  let ready := setLocal env "limit" (.scalar (.int (initialLimit (k+2))))
  obtain ⟨heap, retried⟩ := retryRuntime.retry k (searchLimit (k+2)) (initialLimit (k+2))
    (one_le_initialLimit _) (Nat.le_max_left _ _) (Nat.le_max_right _ _) ready [] rfl rfl rfl
  have calledInitial := EvaluatesIn.call_single (expression := .name "n")
    (env := env) (EvaluatesIn.lookup rfl) (initialRuntime.initial_limit (k+2) [])
  have assigned := BlockReturnsIn.assign (name := "limit") calledInitial retried
  have test : EvaluatesIn .word program env (.compare (.name "n") [(.eq, .constant (.int 1))])
      [] 7 (.scalar (.bool false)) [] :=
    EvaluatesIn.compare (work := 2) (op := .eq) (av := .scalar (.int (k+2 : Nat))) (bv := .scalar (.int 1))
      (EvaluatesIn.lookup rfl) EvaluatesIn.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega) rfl
  have branched := BlockReturnsIn.branch (yes := [.return (.constant (.int 2))]) (no := []) test rfl assigned
  have asserted : EvaluatesIn .word program env (.compare (.name "n") [(.gtE, .constant (.int 1))])
      [] 7 (.scalar (.bool true)) [] :=
    EvaluatesIn.compare (work := 2) (op := .gtE) (av := .scalar (.int (k+2 : Nat))) (bv := .scalar (.int 1))
      (EvaluatesIn.lookup rfl) EvaluatesIn.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega) rfl
  have body := BlockReturnsIn.assertion (message := none) asserted branched
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (values := [.scalar (.int (k+2 : Nat))]) rfl rfl body
  obtain ⟨cost, trace, paid⟩ := called 0 []
  have terminal : RuntimeCost.Finishes (step program 0) (charge .word program)
      ⟨.value (.scalar (.int (Nat.nth Nat.Prime (k+1)))), [], heap⟩ 1
      (.ok ⟨.ok (.scalar (.int (Nat.nth Nat.Prime (k+1)))), [], heap, 0⟩) := .done rfl rfl
  refine ⟨cost+1, _, trace.finishes terminal, ?_⟩
  have initialZero : potential ⟨.operation (.call "solve") [.scalar (.int (k+2 : Nat))], [], []⟩ = 0 := rfl
  rw [initialZero] at paid
  simp only [List.length_cons, List.length_nil] at paid
  have allowance := Nat.mul_le_mul_right (Nat.log2 (searchLimit (k+2))+1)
    (Nat.mul_le_mul_left 1000 (Nat.sub_le (2*searchLimit (k+2)) (initialLimit (k+2))))
  nlinarith only [paid, allowance, Nat.zero_le (potential ⟨.value (.scalar (.int (Nat.nth Nat.Prime (k+1)))), [], heap⟩)]

theorem searchLimit_bound (k : Nat) :
    searchLimit (k+1) ≤ 8*Nat.nth Nat.Prime k*((Nat.nth Nat.Prime k+1).log2+1) := by
  have primeSize := (Nat.prime_nth_prime k).two_le
  have ordinal := Nat.add_two_le_nth_prime k
  have logs := PrimitiveWork.log2_mono (show k+1 ≤ Nat.nth Nat.Prime k+1 by omega)
  have nested := Nat.log2_le_self ((k+1).log2+1)
  have factor : (k+1).log2 + ((k+1).log2+1).log2 + 1 ≤ 2*((Nat.nth Nat.Prime k+1).log2+1) := by omega
  have estimate := Nat.mul_le_mul (show k+1 ≤ Nat.nth Nat.Prime k by omega) factor
  have grows : Nat.nth Nat.Prime k ≤ Nat.nth Nat.Prime k*((Nat.nth Nat.Prime k+1).log2+1) :=
    Nat.le_mul_of_pos_right _ (by omega)
  unfold searchLimit ProjectEulerStatements.P7.naive
  apply max_le
  · unfold initialLimit
    split <;> nlinarith only [estimate, grows, primeSize]
  · nlinarith only [grows]

theorem searchLimit_log_bound (k : Nat) :
    (searchLimit (k+1)).log2+1 ≤ 8*((Nat.nth Nat.Prime k+1).log2+1) := by
  have bound := searchLimit_bound k
  have logBound := Nat.log2_le_self (Nat.nth Nat.Prime k+1)
  have scaled := Nat.mul_le_mul_left (8*Nat.nth Nat.Prime k) (Nat.add_le_add_right logBound 1)
  have polynomial : searchLimit (k+1) ≤ 2^5*(Nat.nth Nat.Prime k+1)^2 := by
    nlinarith only [bound, scaled]
  have width := PrimitiveWork.bitLength_polynomial (Nat.nth Nat.Prime k) (searchLimit (k+1)) 2 5 polynomial
  simp only [PrimitiveWork.bitLength, Int.natAbs_natCast] at width
  omega

theorem prime_bound (k : Nat) :
    TimeBound .word program "solve" [.scalar (.int (k+2 : Nat))] []
      (129000*(Nat.nth Nat.Prime (k+1)*((Nat.nth Nat.Prime (k+1)+1).log2+1)^2)) := by
  apply (source_bound k).mono
  have size := searchLimit_bound (k+1)
  have width := searchLimit_log_bound (k+1)
  have product := Nat.mul_le_mul size width
  have primeSize := (Nat.prime_nth_prime (k+1)).two_le
  have positive : 1 ≤ ((Nat.nth Nat.Prime (k+1)+1).log2+1)^2 := by nlinarith
  have growth := Nat.mul_le_mul_left (Nat.nth Nat.Prime (k+1)) positive
  nlinarith only [product, primeSize, growth]

end solvers.«7».runtimeCost

theorem solvers.«7».solve.«BigO(ProjectEulerStatements.P7.naive n * (Nat.log2 (ProjectEulerStatements.P7.naive n + 1) + 1)^2)» :
    FormalSW.PyAst.HeapCalls.Runtime.IsBigO .word Solvers.P7Ast.module "solve"
      (fun n : Nat => [.scalar (.int n)]) (fun _ => [])
      (fun n : Nat => FormalSW.PyAst.Calls.Condition Solvers.P7Ast.module
        (fun name => if name = "n" then some (.int n) else none) Solvers.P7Ast.requires_solve_0)
      id (fun n : Nat => ProjectEulerStatements.P7.naive n * (Nat.log2 (ProjectEulerStatements.P7.naive n+1)+1)^2) := by
  refine ⟨129000, 2, fun n _ large => ?_⟩
  change 2 ≤ n at large
  rcases n with _ | (_ | k)
  · omega
  · omega
  · simpa only [ProjectEulerStatements.P7.naive] using solvers.«7».runtimeCost.prime_bound k
