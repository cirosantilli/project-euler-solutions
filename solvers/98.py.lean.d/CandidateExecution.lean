import solvers.«98.py.lean.d».CandidatePrepare

namespace solvers.«98».square_search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.AnagramSquares
open pattern_signature (program)
open mapped_word (number integerValue)
open word_patterns (text)
open data.«project-euler-statements».data.py.project_euler_statements.p98.mapping (table)
set_option Elab.async false

def candidateLocals : List String := preparationLocals ++ mapped_word.localNames

/-- Execute the entire candidate body, from fresh bijection tables through every partner word. -/
theorem candidate_result (width : Int) (word square : String) (others : List String) (best : Int)
    (heap : Mutable.Heap) (partners outer recordAddress : Nat) (env : HeapCalls.Env)
    (member : square ∈ candidates.values width.toNat word)
    (valid : ∀ other ∈ others, other.toList ≠ [] ∧ ∀ letter ∈ other.toList, letter ∈ word.toList)
    (foundPartners : heap[partners]? = some (.list (others.map text)))
    (stored : square_cache.Stored outer heap width recordAddress)
    (getWord : env "w1" = some (.scalar (.string word))) (getSquare : env "sq_str" = some (.scalar (.string square)))
    (getPartners : env "others" = some (.ref partners)) (getRecord : env "sqinfo" = some (.ref recordAddress))
    (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, BlockContinues program env bijection.candidateBody heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (mapped_word.scan width.toNat word.toList square.toList others (integerValue square) best)) ∧
      (∀ name, name ∉ candidateLocals → finalEnv name = env name) := by
  obtain ⟨ready, garbage, preparation, getMapping, _, _, getFirst, preparedKept⟩ :=
    prepared width.toNat word square heap env member getWord getSquare
  let allocated := .dict (table (alphabet [] word.toList) (alphabet [] square.toList)) ::
    .dict (table (alphabet [] square.toList) (alphabet [] word.toList)) :: garbage
  obtain ⟨finalEnv, extra, looped, answer, loopKept⟩ := mapped_word.loop_result width word.toList square.toList others
    (integerValue square) best (heap ++ allocated) partners heap.length outer recordAddress ready
    (candidates.matched width.toNat word square member) valid
    (HeapCollection.prefix_found heap allocated partners _ foundPartners) (bijection.found_forward heap garbage _ _)
    (stored.append allocated) ((preparedKept "others" (by decide)).trans getPartners) getMapping
    ((preparedKept "sqinfo" (by decide)).trans getRecord) getFirst ((preparedKept "best" (by decide)).trans getBest)
  refine ⟨finalEnv, allocated ++ extra, ?_, answer, ?_⟩
  · have executed := preparation.append looped
    simpa only [List.take_append_drop, List.append_assoc] using executed
  · intro name absent
    rw [loopKept name (fun member => absent (List.mem_append_right _ member))]
    exact preparedKept name (fun member => absent (List.mem_append_left _ member))

end solvers.«98».square_search
