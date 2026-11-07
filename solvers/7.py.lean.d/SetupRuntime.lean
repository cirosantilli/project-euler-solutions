import solvers.«7.py.lean.d».ScanRuntime
import lang.py.FormalSW.HeapRuntimeCostPrimitives
import solvers.«7.py.lean.d».SieveRuntime

namespace solvers.«7».setupRuntime
open FormalSW.PyAst HeapCalls.Runtime
open scanRuntime (attemptBody)
open ProjectEulerSolutions.P7
private abbrev program := Solvers.P7Ast.module

def initialEnv (env : HeapCalls.Env) (address limit : Nat) : HeapCalls.Env :=
  HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal env
    "size" (.scalar (.int (sieveSize limit)))) "is_prime" (.ref address))
    "root" (.scalar (.int (Nat.sqrt limit))))
    "max_i" (.scalar (.int (Nat.sqrt limit / 2 : Nat)))

def initialHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1]] ++ [.bytearray (sieveBytes (initialSieve limit))]

theorem initialization (limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (.scalar (.int limit))) (mathUnbound : env "math" = none) :
    BlockContinuesIn .word program env (attemptBody.take 5) heap
      (sieveSize limit + Nat.log2 limit + 64)
      (initialEnv env (heap.length + 1) limit) (initialHeap heap limit) := by
  let e1 := HeapCalls.setLocal env "size" (.scalar (.int (sieveSize limit)))
  let e2 := HeapCalls.setLocal e1 "is_prime" (.ref (heap.length + 1))
  let e3 := HeapCalls.setLocal e2 "root" (.scalar (.int (Nat.sqrt limit)))
  let e4 := initialEnv env (heap.length + 1) limit
  let fresh := heap ++ [.bytearray [1]] ++ [.bytearray (List.replicate (sieveSize limit) 1)]
  let heap2 := initialHeap heap limit
  have first : BlockContinuesIn .word program env [attemptBody[0]'(by decide)] heap 13 e1 heap := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (av := .int (limit / 2 : Nat)) (bv := .int 1)
      (ca := 6) (cb := 1) (work := 1)
    · apply EvaluatesIn.binary (av := .int limit) (bv := .int 2) (ca := 1) (cb := 1) (work := 1)
      · exact EvaluatesIn.lookup getLimit
      · exact EvaluatesIn.constant
      · simp [Mutable.binary, scalarBinary]
      · rfl
    · exact EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, sieveSize]
    · rfl
  have second : BlockContinuesIn .word program e1 [attemptBody[1]'(by decide)] heap
      (sieveSize limit + 14) e2 fresh := by
    have repeated := EvaluatesIn.buffer_repeat
      (EvaluatesIn.bytearray (arithmetic := .word) (module := program) (env := e1)
        (heap := heap) (bytes := [1]) EvaluatesIn.constant)
      (EvaluatesIn.lookup (name := "size") (heap := heap ++ [.bytearray [1]]) rfl)
      (by simp : (heap ++ [.bytearray [1]])[heap.length]? = some (.bytearray [1]))
    have assigned := BlockContinuesIn.assign (name := "is_prime") repeated
    simpa [fresh, e2] using assigned.mono (large := sieveSize limit + 14)
      (by simp [PrimitiveWork.integer]; omega)
  have clear : BlockContinuesIn .word program e2 [attemptBody[2]'(by decide)] fresh 14 e2 heap2 := by
    have found : fresh[heap.length+1]? = some (Mutable.Object.bytearray (List.replicate (sieveSize limit) 1)) := by simp [fresh]
    have args : ArgumentsIn .word program e2 0 [.name "is_prime", .constant (.int 0)] fresh 5
        [.ref (heap.length+1), .scalar (.int 0)] fresh :=
      .cons (EvaluatesIn.lookup rfl) (.cons EvaluatesIn.constant .nil)
    have write := Mutable.setItem_buffer fresh (heap.length+1) 0 (List.replicate (sieveSize limit) 1) 0 found
      (by simp [sieveSize])
    have run := BlockContinuesIn.buffer_setItem (arithmetic := .word) (module := program)
      (env := e2) (expression := .constant (.int 0)) (by intros a b c impossible; cases impossible)
      EvaluatesIn.constant args found write
    simpa [fresh, heap2, initialHeap, initialSieve_bytes, PrimitiveWork.integer] using run
  have third : BlockContinuesIn .word program e2 [attemptBody[3]'(by decide)] heap2
      (Nat.log2 limit + 11) e3 heap2 := by
    have argument : EvaluatesIn .word program e2 (.name "limit") heap2 1 (.scalar (.int limit)) heap2 :=
      EvaluatesIn.lookup (by simpa [e2, e1, HeapCalls.setLocal] using getLimit)
    have callee : ReturnsIn .word program "math.isqrt" [.scalar (.int limit)] heap2
        (Nat.log2 limit + 2) (.scalar (.int (Nat.sqrt limit))) heap2 := by
      intro anchors stack
      apply advance (following := ⟨.value (.scalar (.int (Nat.sqrt limit))), stack, heap2⟩)
      · change HeapCalls.result anchors heap2
          (HeapCalls.scalarResult (some (Math.isqrtCall [.int limit]))) stack = _
        have root : Math.isqrt limit = Nat.sqrt limit := Nat.eq_sqrt.mpr (Math.isqrt_spec _)
        simp [HeapCalls.result, HeapCalls.run, HeapCalls.scalarResult, Math.isqrtCall, root,
          show ¬ (limit : Int) < 0 by omega]
      · change some (PrimitiveWork.bitLength (limit : Int) + 1) = _
        simp [PrimitiveWork.bitLength, Nat.add_assoc]
      · rfl
    have called := EvaluatesIn.qualified_single («alias» := "math") (field := "isqrt")
      (by decide) (by decide) (by simpa [e2, e1, HeapCalls.setLocal] using mathUnbound) argument callee
    have converted := EvaluatesIn.builtin_single (builtin := .int)
      (by intro impossible; cases impossible) (by intro impossible; cases impossible) called rfl rfl
    exact (BlockContinuesIn.assign (name := "root") converted).mono (by omega)
  have fourth : BlockContinuesIn .word program e3 [attemptBody[4]'(by decide)] heap2 8 e4 heap2 := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
      (av := .int (Nat.sqrt limit)) (bv := .int 2)
    · exact EvaluatesIn.lookup rfl
    · exact EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary]
    · rfl
  exact (first.append (second.append (clear.append (third.append fourth)))).mono (by omega)

def finalHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1]] ++ [.bytearray (sieveBytes (oddsOnlySieveImpl limit))]

def prepareWork (limit : Nat) : Nat :=
  6*sieveSize limit*((sieveSize limit).log2+2) + 322*sieveSize limit + Nat.log2 limit + 100

theorem prepare (limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (.scalar (.int limit))) (mathUnbound : env "math" = none) :
    ∃ finalEnv, BlockContinuesIn .word program env (attemptBody.take 6) heap
      (prepareWork limit) finalEnv (finalHeap heap limit) ∧
      finalEnv "n" = env "n" ∧ finalEnv "limit" = env "limit" ∧
      finalEnv "size" = some (.scalar (.int (sieveSize limit))) ∧
      finalEnv "is_prime" = some (.ref (heap.length+1)) ∧ finalEnv "math" = none := by
  let initialized := initialEnv env (heap.length+1) limit
  let heap2 := initialHeap heap limit
  have getSize : initialized "size" = some (.scalar (.int (initialSieve limit).size)) := by
    simp [initialized, initialEnv, HeapCalls.setLocal, initialSieve_size]
  have buffer : initialized "is_prime" = some (.ref (heap.length+1)) := rfl
  have found : heap2[heap.length+1]? = some (Mutable.Object.bytearray (sieveBytes (initialSieve limit))) := by
    simp [heap2, initialHeap]
  obtain ⟨finalEnv, loop, stable⟩ := sieveRuntime.loop limit (Nat.sqrt limit / 2) 1 (initialSieve limit)
    initialized heap2 (heap.length+1) (by omega) (by omega)
    (by simpa [initialSieve_size] using maxI_lt_sieveSize limit) getSize buffer found
  have argument : EvaluatesIn .word program initialized
      (.binOp (.name "max_i") .add (.constant (.int 1))) heap2 6
      (.scalar (.int (Nat.sqrt limit / 2 + 1 : Nat))) heap2 := by
    apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
      (av := .int (Nat.sqrt limit / 2 : Nat)) (bv := .int 1)
    · exact EvaluatesIn.lookup rfl
    · exact EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary]
    · rfl
  have ranged := BlockContinuesIn.for_range_pair (a := .constant (.int 1)) EvaluatesIn.constant argument loop
  have initializedRun := initialization limit env heap getLimit mathUnbound
  have combined := initializedRun.append ranged
  refine ⟨finalEnv, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have run : BlockContinuesIn .word program env (attemptBody.take 6) heap
        (sieveSize limit + Nat.log2 limit + 64 +
          (1+6+sieveRuntime.loopWork (sieveSize limit) 1 (Nat.sqrt limit/2+1)+7))
        finalEnv (finalHeap heap limit) := by
      simpa [heap2, initialHeap, finalHeap, initialSieve_size, oddsOnlySieveImpl] using combined
    apply run.mono
    have bound := sieveRuntime.loopWork_bound limit
    unfold prepareWork
    omega
  · simpa [initialized, initialEnv, HeapCalls.setLocal] using stable "n" (by simp)
  · simpa [initialized, initialEnv, HeapCalls.setLocal] using stable "limit" (by simp)
  · simpa [initialized, initialEnv, HeapCalls.setLocal] using stable "size" (by simp)
  · simpa [initialized, initialEnv, HeapCalls.setLocal] using stable "is_prime" (by simp)
  · have equal : finalEnv "math" = env "math" := by
      simpa [initialized, initialEnv, HeapCalls.setLocal] using stable "math" (by simp)
    exact equal.trans mathUnbound

end solvers.«7».setupRuntime
