import solvers.«14.py.lean.d».WalkRuntime
import solvers.«14.py.lean.d».UnwindRuntime
import solvers.«14.py.lean.d».ReturnRuntime
import lang.py.FormalSW.HeapRuntimeCostUnpacking

/-! Compose the actual forward walk, reverse cache fill, and final lookup. -/
namespace solvers.«14».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.HeapCalls FormalSW.PyAst.HeapCalls.Runtime
open ProjectEulerStatements.P14 ProjectEulerStatements.Collatz

theorem collatz_length_returns (limit n smallAddress bigAddress : Nat) (positive : 0 < n)
    (distinct : smallAddress ≠ bigAddress) (base : Mutable.Heap)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : base[smallAddress]? = some (.list small))
    (bigFound : base[bigAddress]? = some (.dict big)) :
    ∃ finalHeap finalSmall finalBig,
      finalHeap[smallAddress]? = some (.list finalSmall) ∧
      finalHeap[bigAddress]? = some (.dict finalBig) ∧
      Cache.Valid limit finalSmall finalBig ∧ Cache.Known limit finalSmall finalBig n ∧
      finalBig.length ≤ big.length + collatzLen n ∧
      ReturnsIn .word program "collatz_length" [.scalar (.int n), .scalar (.int limit), .ref smallAddress, .ref bigAddress]
        base (1000 * (big.length + collatzLen n + 1) * (collatzLen n + 1))
        (.scalar (.int (collatzLen n))) finalHeap := by
  let entry := functionEnv lengthDefinition.1 lengthDefinition.2
    [.scalar (.int n), .scalar (.int limit), .ref smallAddress, .ref bigAddress] []
  let initial := setLocal (setLocal entry "m" (.scalar (.int n))) "path" (.ref base.length)
  have initialLocals : WalkLocals initial n limit smallAddress bigAddress base.length n := by
    constructor <;> rfl
  have setup : BlockContinuesIn .word program entry ((executableBody lengthDefinition.2).take 2) base 8
      initial (base ++ [.list []]) := by
    apply BlockContinuesIn.append (ca := 3) (cb := 5) (body := [.assign [.name "m"] (.name "n")])
    · exact BlockContinuesIn.assign (EvaluatesIn.lookup rfl)
    · exact BlockContinuesIn.annAssign EvaluatesIn.empty_list
  obtain ⟨stop, path, afterWalk, trail, stopPositive, walkLocals, suffixLength, cachedStop, walked⟩ :=
    walk_terminates n limit smallAddress bigAddress base small big valid smallFound bigFound n [] initial positive initialLocals .start
  obtain ⟨finalBase, finalSmall, finalBig, finalEnv, finalSize, finalSmallFound, finalBigFound,
      finalValid, finalKnown, finalLocals, finalLength, growth, unwound⟩ :=
    unwind_prefix n stop path trail limit smallAddress bigAddress stop distinct base small big [] afterWalk
      valid smallFound bigFound walkLocals suffixLength cachedStop
  have smallBound := (List.getElem?_eq_some_iff.mp finalSmallFound).1
  have bigBound := (List.getElem?_eq_some_iff.mp finalBigFound).1
  have smallHere : (finalBase ++ [.list (vertices path)])[smallAddress]? = some (.list finalSmall) := by
    simpa [List.getElem?_append_left smallBound] using finalSmallFound
  have bigHere : (finalBase ++ [.list (vertices path)])[bigAddress]? = some (.dict finalBig) := by
    simpa [List.getElem?_append_left bigBound] using finalBigFound
  have reversed : BlockContinuesIn .word program afterWalk [.forReversed "v" (.name "path") unwindBody []]
      (base ++ [.list (vertices path)])
      (40 * (big.length + path.length + 1) * (path.length + 1) + 5)
      finalEnv (finalBase ++ [.list (vertices path)]) := by
    have run := BlockContinuesIn.for_reversed_list (globals := []) (arithmetic := .word) (module := program)
      (heap := base ++ [.list (vertices path)]) (items := vertices path)
      (EvaluatesIn.lookup walkLocals.path) (by simp)
      (by simpa [vertices] using unwound)
    exact run.mono (by simp only [PrimitiveWork.integer]; omega)
  have pathBound := trail.length_le
  refine ⟨finalBase ++ [.list (vertices path)], finalSmall, finalBig, smallHere, bigHere,
    finalValid, finalKnown, by omega, ?_⟩
  have returned := length_return limit n finalEnv _ smallAddress bigAddress finalSmall finalBig
    finalValid smallHere bigHere finalLocals.small finalLocals.big finalLocals.bound finalLocals.origin finalKnown
  have body := setup.returns (walked.returns (reversed.returns returned))
  have called := BlockReturnsIn.function (name := "collatz_length")
    (args := lengthDefinition.1) (body := lengthDefinition.2)
    (values := [.scalar (.int n), .scalar (.int limit), .ref smallAddress, .ref bigAddress]) rfl rfl body
  apply called.mono
  change 9 + (8 + (500 * (big.length + 1) * (collatzLen n + 1) +
    (40 * (big.length + path.length + 1) * (path.length + 1) + 5 + 50 * (finalBig.length + 1)))) ≤ _
  let work := (big.length + collatzLen n + 1) * (collatzLen n + 1)
  have walkBound : (big.length + 1) * (collatzLen n + 1) ≤ work :=
    Nat.mul_le_mul_right _ (by omega)
  have unwindBound : (big.length + path.length + 1) * (path.length + 1) ≤ work :=
    Nat.mul_le_mul (by omega) (by omega)
  have widthBound : finalBig.length + 1 ≤ work := by
    calc
      finalBig.length + 1 ≤ big.length + collatzLen n + 1 := by omega
      _ ≤ work := Nat.le_mul_of_pos_right _ (by omega)
  have positiveWork : 0 < work := Nat.mul_pos (by omega) (by omega)
  simp only [Nat.mul_assoc]
  change 9 + (8 + (500 * ((big.length + 1) * (collatzLen n + 1)) +
    (40 * ((big.length + path.length + 1) * (path.length + 1)) + 5 + 50 * (finalBig.length + 1)))) ≤ 1000 * work
  omega

end solvers.«14».runtimeCost
