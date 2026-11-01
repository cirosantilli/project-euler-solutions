import solvers.«92.py.lean.d».Terminal
import solvers.«92.py.lean.d».TraversalRow

namespace solvers.«92».weighted
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number := table.number
abbrev condition : Expr :=
  .compare (.call (.name "terminal") [.name "total", .name "known"]) [(.eq, .constant (.int 89))]
abbrev addition : List Stmt := sumBody (.name "ways") []
abbrev selected : List Stmt := [.ifThen condition addition []]
abbrev body : List Stmt := bindings.body ++ selected

/-- The filter and nested binding target are taken from the final source generator. -/
theorem source :
    (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[4]? =
      some (.return (.sumGeneratorTargets (.name "ways")
        [(bindings.target, .call (.attribute (.name "counts") "items") [], [condition])]
        (.constant (.int 0)))) := rfl

def contribution (state : Prefix.State) (ways : Nat) : Nat :=
  if endsAt89 state.1 then ways else 0

def after (locals : HeapCalls.Env) (state : Prefix.State) (ways accumulated : Nat) : HeapCalls.Env :=
  if endsAt89 state.1 then setLocal locals sumAccumulator (number (accumulated + ways)) else locals

/-- The filter runs the real cache-writing helper before deciding whether to add this row. -/
theorem selected_result (heap : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (cache ways accumulated : Nat) (state : Prefix.State) (entries : store.Entries)
    (known : locals "known" = some (.ref cache))
    (total : locals "total" = some (number state.1))
    (multiplicity : locals "ways" = some (number ways))
    (sum : locals sumAccumulator = some (number accumulated))
    (found : store.Found heap cache entries) (valid : Memo.Valid (store.lookup entries)) :
    ∃ result finalHeap,
      BlockContinues program locals selected heap (after locals state ways accumulated) finalHeap globals ∧
      store.Found finalHeap cache result ∧ Memo.Valid (store.lookup result) ∧
      rows.HeapFrame heap finalHeap cache := by
  obtain ⟨value, path, extra, ran, classified, preserved⟩ :=
    terminal.verified state.1 cache entries heap globals found valid
  let result := store.fill entries value path
  let finalHeap := heap.set cache (.dict (store.encode result)) ++ extra
  have filter : Evaluates program locals condition heap (.scalar (.bool (endsAt89 state.1))) finalHeap globals := by
    apply Evaluates.compare (Evaluates.call
      (.cons (Evaluates.lookup total) (.cons (Evaluates.lookup known) .nil)) ran) Evaluates.constant
    have cast : ((value : Int) = 89) ↔ value = 89 := by omega
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, cast, classified, finalHeap, result]
  refine ⟨result, finalHeap, ?_, NaturalDictionary.found_append
    (NaturalDictionary.found_set found result) extra, preserved, ?_⟩
  · apply BlockContinues.branch (flag := endsAt89 state.1) filter
      (by simp [Mutable.truth, scalarTruth])
    cases hit : endsAt89 state.1 with
    | false => simpa [after, hit] using (BlockContinues.nil (module := program)
        (env := locals) (heap := finalHeap) (globals := globals))
    | true =>
        simpa [after, hit, addition, sumBody] using
          (BlockContinues.assign (name := sumAccumulator)
            (Evaluates.binary (Evaluates.lookup sum) (Evaluates.lookup multiplicity)
              (by simp [Mutable.binary, scalarBinary, number, table.number, next_term.number]) :
              Evaluates program locals (.binOp (.name sumAccumulator) .add (.name "ways")) finalHeap
                (number (accumulated + ways)) finalHeap globals))
  · intro other object different present
    have retained : (heap.set cache (.dict (store.encode result)))[other]? = some object :=
      (List.getElem?_set_ne (Ne.symm different)).trans present
    exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp retained).1).trans retained

theorem after_sum (locals : HeapCalls.Env) (state : Prefix.State) (ways accumulated : Nat)
    (sum : locals sumAccumulator = some (number accumulated)) :
    after locals state ways accumulated sumAccumulator = some (number (accumulated + contribution state ways)) := by
  cases hit : endsAt89 state.1 <;> simp [after, contribution, hit, setLocal, sum]

theorem after_other (locals : HeapCalls.Env) (state : Prefix.State) (ways accumulated : Nat)
    (name : String) (different : name ≠ sumAccumulator) :
    after locals state ways accumulated name = locals name := by
  cases hit : endsAt89 state.1 <;> simp [after, hit, setLocal, different]

/-- A yielded pair is unpacked before the filter runs, retaining the shared key identity. -/
theorem row_result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (cache accumulated : Nat) (entry : table.Entry) (entries : store.Entries)
    (known : before "known" = some (.ref cache))
    (sum : before sumAccumulator = some (number accumulated))
    (key : heap[entry.2.1]? = some (table.tuple entry.1))
    (found : store.Found heap cache entries) (valid : Memo.Valid (store.lookup entries)) :
    ∃ result finalHeap finalLocals,
      BlockContinues program (traversal.pairLocals heap before) body (traversal.pairHeap heap entry)
        finalLocals finalHeap globals ∧
      store.Found finalHeap cache result ∧ Memo.Valid (store.lookup result) ∧
      finalLocals "known" = some (.ref cache) ∧
      finalLocals sumAccumulator = some (number (accumulated + contribution entry.1 entry.2.2)) ∧
      rows.HeapFrame heap finalHeap cache := by
  let unpacked := bindings.locals (traversal.pairLocals heap before) entry.2.1 entry.2.2 entry.1
  have pair : (traversal.pairHeap heap entry)[heap.length]? =
      some (.tuple [.ref entry.2.1, number entry.2.2]) := by simp [traversal.pairHeap]
  have saved : (traversal.pairHeap heap entry)[entry.2.1]? = some (table.tuple entry.1) :=
    (List.getElem?_append_left (List.getElem?_eq_some_iff.mp key).1).trans key
  have bound := bindings.result (traversal.pairHeap heap entry) (traversal.pairLocals heap before) globals
    heap.length entry.2.1 entry.2.2 entry.1 (by simp [traversal.pairLocals, setLocal]) pair saved
  have getKnown : unpacked "known" = some (.ref cache) := by
    simpa [unpacked, bindings.locals, assignNames, traversal.pairLocals, setLocal] using known
  have getSum : unpacked sumAccumulator = some (number accumulated) := by
    simpa [unpacked, bindings.locals, assignNames, traversal.pairLocals, setLocal, sumAccumulator] using sum
  obtain ⟨result, finalHeap, ran, finalFound, finalValid, frame⟩ :=
    selected_result (traversal.pairHeap heap entry) unpacked globals cache entry.2.2 accumulated entry.1 entries
      getKnown (by simp [unpacked, bindings.locals, assignNames, setLocal])
      (by simp [unpacked, bindings.locals, assignNames, setLocal]) getSum
      (NaturalDictionary.found_append found _) valid
  refine ⟨result, finalHeap, after unpacked entry.1 entry.2.2 accumulated,
    bound.append ran, finalFound, finalValid, ?_, after_sum _ _ _ _ getSum, ?_⟩
  · rw [after_other _ _ _ _ _ (by decide)]
    exact getKnown
  · intro other object different present
    exact frame other object different
      ((List.getElem?_append_left (List.getElem?_eq_some_iff.mp present).1).trans present)

/-- Updating the endpoint cache cannot alter any tuple key in the counts dictionary. -/
theorem keys_frame (before after : Mutable.Heap) (cache : Nat) (entries : table.Entries)
    (cached : store.Entries) (keys : table.Keys before entries) (found : store.Found before cache cached)
    (frame : rows.HeapFrame before after cache) : table.Keys after entries := by
  intro entry member
  have key := keys entry member
  have different : entry.2.1 ≠ cache := by
    intro same
    rw [same, found] at key
    simp [table.tuple] at key
  exact frame _ _ different key

end solvers.«92».weighted
