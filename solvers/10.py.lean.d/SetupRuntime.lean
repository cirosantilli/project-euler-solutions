import solvers.«10.py.lean.d».SumRuntime
import solvers.«10.py.lean.d».SieveRuntime

namespace solvers.«10».setupRuntime
open FormalSW.PyAst HeapCalls.Runtime
open sumRuntime (solveBody)
private abbrev program := Solvers.P10Ast.module

def initialEnv (env : HeapCalls.Env) (address limit : Nat) : HeapCalls.Env :=
  HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal env
    "size" (.scalar (.int (sieveSize limit)))) "sieve" (.ref address))
    "r" (.scalar (.int (Nat.sqrt (limit - 1)))))
    "max_i" (.scalar (.int (((Nat.sqrt (limit - 1) : Int) - 3) / 2)))

def initialHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1]] ++ [.bytearray (List.replicate (sieveSize limit) 1)]

theorem initialization (limit : Nat) (large : 4 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (.scalar (.int limit))) (mathUnbound : env "math" = none) :
    BlockContinuesIn .word program env ((solveBody.drop 2).take 4) heap
      (sieveSize limit + Nat.log2 (limit - 1) + 52)
      (initialEnv env (heap.length + 1) limit) (initialHeap heap limit) := by
  let e1 := HeapCalls.setLocal env "size" (.scalar (.int (sieveSize limit)))
  let e2 := HeapCalls.setLocal e1 "sieve" (.ref (heap.length + 1))
  let e3 := HeapCalls.setLocal e2 "r" (.scalar (.int (Nat.sqrt (limit - 1))))
  let e4 := initialEnv env (heap.length + 1) limit
  let heap2 := initialHeap heap limit
  have first : BlockContinuesIn .word program env [solveBody[2]'(by decide)] heap 13 e1 heap := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (av := .int (limit / 2 : Nat)) (bv := .int 1)
      (ca := 6) (cb := 1) (work := 1)
    · apply EvaluatesIn.binary (av := .int limit) (bv := .int 2) (ca := 1) (cb := 1) (work := 1)
      · exact EvaluatesIn.lookup getLimit
      · exact EvaluatesIn.constant
      · simp [Mutable.binary, scalarBinary]
      · rfl
    · exact EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, sieveSize_cast limit large]
    · rfl
  have second : BlockContinuesIn .word program e1 [solveBody[3]'(by decide)] heap
      (sieveSize limit + 14) e2 heap2 := by
    have repeated := EvaluatesIn.buffer_repeat
      (EvaluatesIn.bytearray (arithmetic := .word) (module := program) (env := e1)
        (heap := heap) (bytes := [1]) EvaluatesIn.constant)
      (EvaluatesIn.lookup (name := "size") (heap := heap ++ [.bytearray [1]]) rfl)
      (by simp : (heap ++ [.bytearray [1]])[heap.length]? = some (.bytearray [1]))
    have assigned := BlockContinuesIn.assign (name := "sieve") repeated
    simpa [heap2, initialHeap, e2] using assigned.mono (large := sieveSize limit + 14)
      (by simp [PrimitiveWork.integer]; omega)
  have third : BlockContinuesIn .word program e2 [solveBody[4]'(by decide)] heap2
      (Nat.log2 (limit - 1) + 12) e3 heap2 := by
    have argument : EvaluatesIn .word program e2 (.binOp (.name "limit") .sub (.constant (.int 1)))
        heap2 6 (.scalar (.int (limit - 1 : Nat))) heap2 := by
      apply EvaluatesIn.binary (av := .int limit) (bv := .int 1) (ca := 1) (cb := 1) (work := 1)
      · exact EvaluatesIn.lookup (by simpa [e2, e1, HeapCalls.setLocal] using getLimit)
      · exact EvaluatesIn.constant
      · simp [Mutable.binary, scalarBinary, Int.natCast_sub (show 1 ≤ limit by omega)]
      · rfl
    have callee : ReturnsIn .word program "math.isqrt" [.scalar (.int (limit - 1 : Nat))] heap2
        (Nat.log2 (limit - 1) + 2) (.scalar (.int (Nat.sqrt (limit - 1)))) heap2 := by
      intro anchors stack
      apply advance (following := ⟨.value (.scalar (.int (Nat.sqrt (limit - 1)))), stack, heap2⟩)
      · change HeapCalls.result anchors heap2
          (HeapCalls.scalarResult (some (Math.isqrtCall [.int (limit - 1 : Nat)]))) stack = _
        have root : Math.isqrt (limit - 1) = Nat.sqrt (limit - 1) := Nat.eq_sqrt.mpr (Math.isqrt_spec _)
        simp [HeapCalls.result, HeapCalls.run, HeapCalls.scalarResult, Math.isqrtCall, root,
          show ¬ ((limit - 1 : Nat) : Int) < 0 by omega]
      · change some (PrimitiveWork.bitLength ((limit - 1 : Nat) : Int) + 1) = _
        simp [PrimitiveWork.bitLength, Nat.add_assoc]
      · rfl
    have called := EvaluatesIn.qualified_single («alias» := "math") (field := "isqrt")
      (by decide) (by decide) (by simpa [e2, e1, HeapCalls.setLocal] using mathUnbound) argument callee
    exact (BlockContinuesIn.assign (name := "r") called).mono (by omega)
  have fourth : BlockContinuesIn .word program e3 [solveBody[5]'(by decide)] heap2 13 e4 heap2 := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (ca := 6) (cb := 1) (work := 1)
      (av := .int ((Nat.sqrt (limit - 1) : Int) - 3)) (bv := .int 2)
    · apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
        (av := .int (Nat.sqrt (limit - 1))) (bv := .int 3)
      · exact EvaluatesIn.lookup rfl
      · exact EvaluatesIn.constant
      · rfl
      · rfl
    · exact EvaluatesIn.constant
    · rfl
    · rfl
  exact (first.append (second.append (third.append fourth))).mono (by omega)

private theorem for_range_single (argument : EvaluatesIn .word module env expression heap ca (.scalar (.int stop)) middle)
    (body : BlockContinuesIn .word module env [.rangeStepLoop name 0 stop 1 loop otherwise]
      middle budget finalEnv finalHeap) :
    BlockContinuesIn .word module env [.forIn (.name name) (.call (.name "range") [expression]) loop otherwise]
      heap (ca + budget + 4) finalEnv finalHeap := by
  intro following anchors stack
  have initial : Reaches .word module anchors
      ⟨.block ([.forIn (.name name) (.call (.name "range") [expression]) loop otherwise] ++ following) env, stack, heap⟩ 1
      ⟨.expr expression env, .rangeArguments name [] [] loop otherwise following env :: stack, heap⟩ := advance rfl rfl rfl
  have ready : Reaches .word module anchors
      ⟨.value (.scalar (.int stop)), .rangeArguments name [] [] loop otherwise following env :: stack, middle⟩ 1
      ⟨.startRange name [.scalar (.int stop)] loop otherwise following env, stack, middle⟩ := advance rfl rfl rfl
  have enter : Reaches .word module anchors
      ⟨.startRange name [.scalar (.int stop)] loop otherwise following env, stack, middle⟩ 2
      ⟨.block ([.rangeStepLoop name 0 stop 1 loop otherwise] ++ following) env, stack, middle⟩ := advance rfl rfl rfl
  exact ((((initial.trans (argument anchors _)).trans ready).trans enter).trans (body following anchors stack)).mono (by omega)

theorem prepare (limit : Nat) (large : 4 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (.scalar (.int limit))) (mathUnbound : env "math" = none) :
    ∃ finalEnv finalHeap finalBytes,
      BlockContinuesIn .word program env ((solveBody.drop 2).take 5) heap
        (6 * sieveSize limit * ((sieveSize limit).log2 + 2) + 317 * sieveSize limit + Nat.log2 (limit - 1) + 66)
        finalEnv finalHeap ∧
      finalEnv "sieve" = some (.ref (heap.length + 1)) ∧
      finalHeap[heap.length + 1]? = some (.bytearray finalBytes) ∧ finalBytes.length = sieveSize limit := by
  let initialized := initialEnv env (heap.length + 1) limit
  let heap2 := initialHeap heap limit
  have getSize : initialized "size" = some (.scalar (.int (List.replicate (sieveSize limit) (1 : UInt8)).length)) := by
    simp [initialized, initialEnv, HeapCalls.setLocal]
  have getSieve : initialized "sieve" = some (.ref (heap.length + 1)) := rfl
  have found : heap2[heap.length + 1]? = some (.bytearray (List.replicate (sieveSize limit) 1)) := by
    simp [heap2, initialHeap]
  obtain ⟨finalEnv, finalHeap, finalBytes, loop, finalFound, finalLength, _, finalSieve⟩ :=
    sieveRuntime.loop limit 0 (List.replicate (sieveSize limit) 1) initialized heap2 (heap.length + 1)
      (by omega) (by simp) getSize getSieve found
  have ranged : BlockContinuesIn .word program initialized [solveBody[6]'(by decide)] heap2
      (sieveRuntime.loopWork (sieveSize limit) 0 ((Nat.sqrt (limit - 1) - 1) / 2) + 10) finalEnv finalHeap := by
    have argument : EvaluatesIn .word program initialized
        (.binOp (.name "max_i") .add (.constant (.int 1))) heap2 6
        (.scalar (.int ((Nat.sqrt (limit - 1) - 1) / 2 : Nat))) heap2 := by
      apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
        (av := .int (((Nat.sqrt (limit - 1) : Int) - 3) / 2)) (bv := .int 1)
      · exact EvaluatesIn.lookup rfl
      · exact EvaluatesIn.constant
      · change (some (.ok (.scalar (.int (((Nat.sqrt (limit - 1) : Int) - 3) / 2 + 1))), heap2) :
          Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
        rw [candidate_count _ (Nat.sqrt_pos.mpr (by omega))]
      · rfl
    have run := for_range_single (name := "i") argument loop
    simpa only [List.length_replicate] using run.mono
      (large := sieveRuntime.loopWork (sieveSize limit) 0 ((Nat.sqrt (limit - 1) - 1) / 2) + 10)
      (by simp; omega)
  refine ⟨finalEnv, finalHeap, finalBytes, ?_, finalSieve.trans getSieve, finalFound, ?_⟩
  · apply ((initialization limit large env heap getLimit mathUnbound).append ranged).mono
    have upper := sieveRuntime.loopWork_bound (sieveSize limit) ((Nat.sqrt (limit - 1) - 1) / 2) (by
      have root := Nat.sqrt_le_self (limit - 1)
      unfold sieveSize
      omega)
    omega
  · simpa using finalLength

end solvers.«10».setupRuntime
