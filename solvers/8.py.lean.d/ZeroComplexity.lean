import «.formalsw».solvers.«8.py.ast»
import lang.py.FormalSW.AllocationCost

/-! A fixed-width, unbounded input family exhibiting the copying improvement:
all-zero lists of arbitrary length. Every scalar/heap transition is counted. -/
namespace solvers.«8».complexity
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open AllocationCost
set_option maxRecDepth 4096
set_option maxHeartbeats 1000000

private abbrev program := Solvers.P8Ast.module
private def zeros (n : Nat) : Heap := [.list (List.replicate n (.scalar (.int 0)))]
private def body : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private def outer : Stmt := body[4]?.getD (.unsupported "missing outer loop")
private def outerBody : List Stmt := match outer with | .whileDo _ body _ => body | _ => []
private def scan : Stmt := outerBody[0]?.getD (.unsupported "missing zero scan")
private def rest : List Stmt := outerBody.drop 1 ++ [Calls.loopContinuation outerBody outer] ++ body.drop 5
private def initialEnv : HeapCalls.Env := HeapCalls.argumentEnv
  [⟨"digits", some (.subscript (.name "list") (.name "int")), none⟩, ⟨"k", some (.name "int"), none⟩]
  [.ref 0, .scalar (.int 1)]
private def scanEnv (n i : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal initialEnv "digit_count" (.scalar (.int n))) "best" (.scalar (.int 0))) "i" (.scalar (.int i))

private theorem enter_scan (n : Nat) (positive : 0 < n) :
    ReachesWith (step program 1) (charge program 1)
      ⟨.operation (.call "solve") [.ref 0, .scalar (.int 1)], [], zeros n⟩ 27
      ⟨.block (scan :: rest) (scanEnv n 0), [], zeros n⟩ := by
  repeat first | exact ReachesWith.refl | refine next_same_heap rfl rfl ?_
  try simp only [List.length_replicate]
  repeat first | exact ReachesWith.refl | refine next_same_heap rfl rfl ?_
  try simp only [evalCmpOp, Int.natCast_pos.mpr positive, decide_true]
  repeat first | exact ReachesWith.refl | refine next_same_heap rfl rfl ?_

private theorem read_zero (n i : Nat) (within : i < n) :
    Mutable.getItem (.ref 0) (.scalar (.int i)) (zeros n) =
      some (.ok (.scalar (.int 0)), zeros n) := by
  have nonnegative : ¬ (i : Int) < 0 := by omega
  simp [Mutable.getItem, Mutable.index, zeros, nonnegative, Nat.not_le.mpr within, List.getElem?_replicate_of_lt within, Except.map]

private def scanStack (n i : Nat) : List Frame :=
  [.compareLeft .eq (.constant (.int 0)) (scanEnv n i),
   .boolean .and [] (scanEnv n i),
   .branch ([.augAssign (.name "i") .add (.constant (.int 1))] ++
     [Calls.loopContinuation [.augAssign (.name "i") .add (.constant (.int 1))] scan])
     [] rest (scanEnv n i)]

private theorem scan_step (n i : Nat) (within : i < n) :
    ReachesWith (step program 1) (charge program 1)
      ⟨.block (scan :: rest) (scanEnv n i), [], zeros n⟩ 26
      ⟨.block (scan :: rest) (scanEnv n (i + 1)), [], zeros n⟩ := by
  repeat first | exact ReachesWith.refl | refine next_same_heap rfl rfl ?_
  simp only [evalCmpOp, Int.ofNat_lt.mpr within, decide_true]
  repeat first | exact ReachesWith.refl | refine next_same_heap rfl rfl ?_
  refine next_same_heap (following := ⟨.value (.scalar (.int 0)), scanStack n i, zeros n⟩) ?_ rfl ?_
  · change protectGenerator _ (result 1 (zeros n) (Mutable.getItem (.ref 0) (.scalar (.int i))) _) = _
    simp only [result, run, read_zero n i within, protectGenerator_next]
    rfl
  repeat first | exact ReachesWith.refl | refine next_same_heap rfl rfl ?_
  try simp only [Int.natCast_add, Int.natCast_one]
  have shadow (env : HeapCalls.Env) (a b : Mutable.Value) : setLocal (setLocal env "i" a) "i" b = setLocal env "i" b := by
    funext name
    by_cases name = "i" <;> simp [setLocal, *]
  simp only [scanEnv, shadow]
  exact .refl

private theorem scan_done (n : Nat) :
    ReachesWith (step program 1) (charge program 1)
      ⟨.block (scan :: rest) (scanEnv n n), [], zeros n⟩ 9
      ⟨.block rest (scanEnv n n), [], zeros n⟩ := by
  repeat first | exact ReachesWith.refl | refine next_same_heap rfl rfl ?_
  simp only [evalCmpOp, Int.lt_irrefl, decide_false]
  repeat first | exact ReachesWith.refl | refine next_same_heap rfl rfl ?_

private theorem scan_all (n remaining i : Nat) (total : i + remaining = n) :
    ReachesWith (step program 1) (charge program 1)
      ⟨.block (scan :: rest) (scanEnv n i), [], zeros n⟩ (26 * remaining + 9)
      ⟨.block rest (scanEnv n n), [], zeros n⟩ := by
  induction remaining generalizing i with
  | zero => simp only [Nat.add_zero] at total; subst i; exact scan_done n
  | succ remaining ih =>
    have next := (scan_step n i (by omega)).trans (ih (i+1) (by omega))
    simpa [Nat.mul_add, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using next

private theorem finish_rest (n : Nat) :
    ReachesWith (step program 1) (charge program 1)
      ⟨.block rest (scanEnv n n), [], zeros n⟩ 10
      ⟨.value (.scalar (.int 0)), [], zeros n⟩ := by
  repeat first | exact ReachesWith.refl | refine next_same_heap rfl rfl ?_
  simp only [evalCmpOp, Int.le_refl, decide_true]
  repeat first | exact ReachesWith.refl | refine next_same_heap rfl rfl ?_

def optimizedZeros (n : Nat) : Execution Mutable.Completion :=
  (HeapCalls.call program "solve" [.ref 0, .scalar (.int 1)] (zeros n)).execution

def zeroWeights (n : Nat) : (optimizedZeros n).State → Nat := charge program 1

/-- An unbounded fixed-width input family: each zero is inspected once and no
list is copied. Includes the terminal transition. -/
theorem optimized_zeros_exact (n : Nat) (positive : 0 < n) :
    FinishesWith (optimizedZeros n).step (zeroWeights n) (optimizedZeros n).initial
      (26*n + 47) (.ok ⟨.ok (.scalar (.int 0)), [], zeros n, 1⟩) := by
  have run := ((enter_scan n positive).trans (scan_all n n 0 (by omega))).trans (finish_rest n)
  have finished := run.finishes (finish_value program 1 (.scalar (.int 0)) (zeros n))
  have same : 27 + (26*n+9) + 10 + 1 = 26*n+47 := by omega
  rw [same] at finished
  exact finished

theorem optimized_zeros_bigO : Execution.WeightedBigO optimizedZeros zeroWeights
    (fun _ => True) id id := by
  refine ⟨73, 1, ?_⟩
  intro n _ hn
  change 1 ≤ n at hn
  refine ⟨26*n+47, _, optimized_zeros_exact n hn, ?_⟩
  change 26*n+47 ≤ 73*n
  omega

end solvers.«8».complexity

