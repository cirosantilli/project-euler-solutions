import solvers.«9.py.lean.d».OuterRuntime
import lang.py.FormalSW.DecimalWorkBounds

namespace solvers.«9».runtimeCost
open FormalSW.PyAst HeapCalls.Runtime solvers.«9».runtimeSource
private abbrev program := Solvers.P9Ast.module

private theorem message_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total : Nat)
    (ht : env "total" = some (.scalar (.int total))) :
    EvaluatesIn .word program env errorMessage heap (2 * total + 100)
      (.scalar (.string (("" ++ "No Pythagorean triplet sums to ") ++ toString (total : Int)))) heap := by
  have converted : EvaluatesIn .word program env (.builtinCall .str [.name "total"]) heap
      (1 + (PrimitiveWork.bitLength (total : Int) + 1) + 3) (.scalar (.string (toString (total : Int)))) heap :=
    EvaluatesIn.builtin_single (by intro h; cases h) (by intro h; cases h) (EvaluatesIn.lookup ht) rfl rfl
  have leading : EvaluatesIn .word program env
      (.binOp (.constant (.string "")) .add (.constant (.string "No Pythagorean triplet sums to "))) heap 37
      (.scalar (.string ("" ++ "No Pythagorean triplet sums to "))) heap :=
    EvaluatesIn.binary (work := 32) EvaluatesIn.constant EvaluatesIn.constant rfl rfl
  have message := EvaluatesIn.binary (op := .add) (work := 32 + (toString (total : Int)).length) leading converted rfl
    (by
      simp [PrimitiveWork.binary, scalarNumber]
      change 31 + (toString (total : Int)).length + 1 = 32 + (toString (total : Int)).length
      omega)
  apply message.mono
  have digits := PrimitiveWork.decimal_length total (total + 1) (by omega)
    (Nat.lt_trans (Nat.lt_succ_self total) (Nat.lt_pow_self (by decide : 1 < 10)))
  have logarithm := Nat.log2_le_self total
  simp only [PrimitiveWork.bitLength, Int.natAbs_natCast]
  omega

/-- Both exits are charged. Formatting the failure message is bounded as well,
so this runtime bound does not need to assume a triplet exists. -/
theorem functionTail_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total : Nat) (best : Int)
    (ht : env "total" = some (.scalar (.int total))) (hb : env "best" = some (.scalar (.int best)))
    (anchors : Nat) :
    ∃ cost outcome, cost ≤ 2 * total + 110 ∧ RuntimeCost.Finishes (HeapCalls.step program anchors)
      (charge .word program) ⟨.block functionTail env, [], heap⟩ cost outcome := by
  by_cases zero : best = 0
  · have test : EvaluatesIn .word program env (.name "best") heap 1 (.scalar (.int 0)) heap := by
      simpa only [zero] using (EvaluatesIn.lookup (arithmetic := .word) (module := program) (heap := heap) hb)
    have branch : BlockContinuesIn .word program env [.ifThen (.name "best") [.return (.name "best")] []]
        heap 3 env heap := BlockContinuesIn.branch test rfl BlockContinuesIn.nil
    obtain ⟨messageCost, message, messageBound⟩ := message_cost env heap total ht anchors
      [.arguments (.raiseException "ValueError") [] [] env]
    have raised : RuntimeCost.Finishes (HeapCalls.step program anchors) (charge .word program)
        ⟨.block [raiseError] env, [], heap⟩ (messageCost + 5)
        (.ok ⟨.error {typeName := "ValueError", args := [.string (("" ++ "No Pythagorean triplet sums to ") ++ toString (total : Int))]},
          [], heap, anchors⟩) := by
      have tail : RuntimeCost.Finishes (HeapCalls.step program anchors) (charge .word program)
          ⟨.value (.scalar (.string (("" ++ "No Pythagorean triplet sums to ") ++ toString (total : Int)))),
            [.arguments (.raiseException "ValueError") [] [] env], heap⟩ 4
          (.ok ⟨.error {typeName := "ValueError", args := [.string (("" ++ "No Pythagorean triplet sums to ") ++ toString (total : Int))]},
            [], heap, anchors⟩) := .next (amount := 1) (cost := 3) rfl rfl (.done rfl rfl)
      have run := RuntimeCost.Finishes.next (step := HeapCalls.step program anchors) (amount := 1) (charge := charge .word program)
        (state := ⟨.block [raiseError] env, [], heap⟩) rfl rfl (message.finishes tail)
      simpa only [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using run
    obtain ⟨branchCost, branched, branchBound⟩ := branch [raiseError] anchors []
    refine ⟨branchCost + (messageCost + 5), _, ?_, branched.finishes raised⟩
    simp only [potential] at messageBound branchBound
    omega
  · have returned : BlockReturnsIn .word program env functionTail heap 5 (.scalar (.int best)) heap := by
      apply BlockReturnsIn.branch (cc := 1) (cb := 2) (flag := true) (EvaluatesIn.lookup hb)
      · simp [Mutable.truth, scalarTruth, zero]
      · exact BlockReturnsIn.return_value (EvaluatesIn.lookup hb)
    obtain ⟨cost, trace, paid⟩ := returned anchors []
    refine ⟨cost + 1, _, ?_, trace.finishes (.done rfl rfl)⟩
    simp only [potential] at paid
    omega

end solvers.«9».runtimeCost
