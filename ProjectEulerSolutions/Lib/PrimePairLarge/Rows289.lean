import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9371_small : List Nat := []
theorem row9371_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9371 row9371_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9371_block_005 : List Nat := [9467, 9491, 9497, 10337]
def plan9371_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .possible, .possible, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), (.secondFactor 31), (.secondFactor 37), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 37), (.firstFactor 7), .first, .second, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .possible, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row9371_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9371 row9371_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9371_block_005
  · decide
  · rfl

def row9371_block_006 : List Nat := [10859, 11657, 11699, 12269]
def plan9371_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 23), .second, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 17), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .second, .first, .second, (.firstFactor 7), .first, (.secondFactor 13), .first, .second, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9371_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9371 row9371_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9371_block_006
  · decide
  · rfl

def row9371_block_007 : List Nat := [12809, 13007, 13613]
def plan9371_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 17), (.firstFactor 37), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 17), .first, .first, .first, (.secondFactor 19), (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row9371_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9371 row9371_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9371_block_007
  · decide
  · rfl

def row9371_block_008 : List Nat := [14639, 15149, 15161, 16001, 16487]
def plan9371_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 17), .second, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .possible, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 37), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .possible, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 7), .possible, (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row9371_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9371 row9371_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9371_block_008
  · decide
  · rfl

def row9371_block_009 : List Nat := [16547]
def plan9371_block_009 : List Budget.Choice := [.first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9371_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9371 row9371_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9371_block_009
  · decide
  · rfl

def row9371_block_010 : List Nat := []
theorem row9371_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9371 row9371_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9371_block_011 : List Nat := []
theorem row9371_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9371 row9371_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9371_block_012 : List Nat := []
theorem row9371_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9371 row9371_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9371_block_013 : List Nat := []
theorem row9371_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9371 row9371_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9371_large_left_left : List Nat := row9371_small ++ row9371_block_005
theorem row9371_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9371 row9371_large_left_left :=
  Budget.RowUpperBound.branch row9371_small_covers row9371_block_005_covers

def row9371_large_left_right_right : List Nat := row9371_block_007 ++ row9371_block_008
theorem row9371_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9371 row9371_large_left_right_right :=
  Budget.RowUpperBound.branch row9371_block_007_covers row9371_block_008_covers

def row9371_large_left_right : List Nat := row9371_block_006 ++ row9371_large_left_right_right
theorem row9371_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9371 row9371_large_left_right :=
  Budget.RowUpperBound.branch row9371_block_006_covers row9371_large_left_right_right_covers

def row9371_large_left : List Nat := row9371_large_left_left ++ row9371_large_left_right
theorem row9371_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9371 row9371_large_left :=
  Budget.RowUpperBound.branch row9371_large_left_left_covers row9371_large_left_right_covers

def row9371_large_right_left : List Nat := row9371_block_009 ++ row9371_block_010
theorem row9371_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9371 row9371_large_right_left :=
  Budget.RowUpperBound.branch row9371_block_009_covers row9371_block_010_covers

def row9371_large_right_right_right : List Nat := row9371_block_012 ++ row9371_block_013
theorem row9371_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9371 row9371_large_right_right_right :=
  Budget.RowUpperBound.branch row9371_block_012_covers row9371_block_013_covers

def row9371_large_right_right : List Nat := row9371_block_011 ++ row9371_large_right_right_right
theorem row9371_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9371 row9371_large_right_right :=
  Budget.RowUpperBound.branch row9371_block_011_covers row9371_large_right_right_right_covers

def row9371_large_right : List Nat := row9371_large_right_left ++ row9371_large_right_right
theorem row9371_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9371 row9371_large_right :=
  Budget.RowUpperBound.branch row9371_large_right_left_covers row9371_large_right_right_covers

def row9371_large : List Nat := row9371_large_left ++ row9371_large_right
theorem row9371_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9371 row9371_large :=
  Budget.RowUpperBound.branch row9371_large_left_covers row9371_large_right_covers

def row9371 : List Nat := row9371_large
theorem row9371_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9371 row9371 := row9371_large_covers

def row9377_small : List Nat := []
theorem row9377_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9377 row9377_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9377_block_005 : List Nat := [9461, 9629, 9767, 10133]
def plan9377_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .possible, .first, (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 19), (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9377_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9377 row9377_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9377_block_005
  · decide
  · rfl

def row9377_block_006 : List Nat := [10601, 10883, 11321, 11657, 11813, 11939]
def plan9377_block_006 : List Budget.Choice := [(.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 29), .second, .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .second, (.firstFactor 17), .first, .first, .first, .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, (.secondFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.secondFactor 23), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 23), .possible, .first, (.secondFactor 31), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 29), .second, (.firstFactor 23), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .second, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 17), .second, .first, (.secondFactor 11), (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9377_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9377 row9377_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9377_block_006
  · decide
  · rfl

def row9377_block_007 : List Nat := [12527]
def plan9377_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 37), (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 19), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .second, .first, (.secondFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9377_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9377 row9377_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9377_block_007
  · decide
  · rfl

def row9377_block_008 : List Nat := [14891, 16103, 16349]
def plan9377_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .second, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .second, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .second, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .possible, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 23), (.secondFactor 11), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9377_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9377 row9377_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9377_block_008
  · decide
  · rfl

def row9377_block_009 : List Nat := [16529]
def plan9377_block_009 : List Budget.Choice := [.first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9377_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9377 row9377_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9377_block_009
  · decide
  · rfl

def row9377_block_010 : List Nat := []
theorem row9377_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9377 row9377_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9377_block_011 : List Nat := []
theorem row9377_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9377 row9377_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9377_block_012 : List Nat := []
theorem row9377_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9377 row9377_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9377_block_013 : List Nat := []
theorem row9377_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9377 row9377_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9377_large_left_left : List Nat := row9377_small ++ row9377_block_005
theorem row9377_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9377 row9377_large_left_left :=
  Budget.RowUpperBound.branch row9377_small_covers row9377_block_005_covers

def row9377_large_left_right_right : List Nat := row9377_block_007 ++ row9377_block_008
theorem row9377_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9377 row9377_large_left_right_right :=
  Budget.RowUpperBound.branch row9377_block_007_covers row9377_block_008_covers

def row9377_large_left_right : List Nat := row9377_block_006 ++ row9377_large_left_right_right
theorem row9377_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9377 row9377_large_left_right :=
  Budget.RowUpperBound.branch row9377_block_006_covers row9377_large_left_right_right_covers

def row9377_large_left : List Nat := row9377_large_left_left ++ row9377_large_left_right
theorem row9377_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9377 row9377_large_left :=
  Budget.RowUpperBound.branch row9377_large_left_left_covers row9377_large_left_right_covers

def row9377_large_right_left : List Nat := row9377_block_009 ++ row9377_block_010
theorem row9377_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9377 row9377_large_right_left :=
  Budget.RowUpperBound.branch row9377_block_009_covers row9377_block_010_covers

def row9377_large_right_right_right : List Nat := row9377_block_012 ++ row9377_block_013
theorem row9377_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9377 row9377_large_right_right_right :=
  Budget.RowUpperBound.branch row9377_block_012_covers row9377_block_013_covers

def row9377_large_right_right : List Nat := row9377_block_011 ++ row9377_large_right_right_right
theorem row9377_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9377 row9377_large_right_right :=
  Budget.RowUpperBound.branch row9377_block_011_covers row9377_large_right_right_right_covers

def row9377_large_right : List Nat := row9377_large_right_left ++ row9377_large_right_right
theorem row9377_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9377 row9377_large_right :=
  Budget.RowUpperBound.branch row9377_large_right_left_covers row9377_large_right_right_covers

def row9377_large : List Nat := row9377_large_left ++ row9377_large_right
theorem row9377_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9377 row9377_large :=
  Budget.RowUpperBound.branch row9377_large_left_covers row9377_large_right_covers

def row9377 : List Nat := row9377_large
theorem row9377_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9377 row9377 := row9377_large_covers

def row9391_small : List Nat := []
theorem row9391_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9391 row9391_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9391_block_005 : List Nat := [9613, 9811]
def plan9391_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .possible, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 17), (.firstFactor 17), (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 17), .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .second]
set_option maxRecDepth 2048 in
theorem row9391_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9391 row9391_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9391_block_005
  · decide
  · rfl

def row9391_block_006 : List Nat := [10867, 10987, 12073]
def plan9391_block_006 : List Budget.Choice := [.first, .second, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 17), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .possible, .first, .first, (.firstFactor 19), (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .possible, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .second, (.secondFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .second, .first, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row9391_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9391 row9391_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9391_block_006
  · decide
  · rfl

def row9391_block_007 : List Nat := []
def plan9391_block_007 : List Budget.Choice := [(.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 31), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9391_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9391 row9391_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9391_block_007
  · decide
  · rfl

def row9391_block_008 : List Nat := [14821, 15187, 15361, 15559, 16417]
def plan9391_block_008 : List Budget.Choice := [(.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 37), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .second, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9391_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9391 row9391_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9391_block_008
  · decide
  · rfl

def row9391_block_009 : List Nat := []
def plan9391_block_009 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9391_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9391 row9391_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9391_block_009
  · decide
  · rfl

def row9391_block_010 : List Nat := []
theorem row9391_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9391 row9391_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9391_block_011 : List Nat := []
theorem row9391_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9391 row9391_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9391_block_012 : List Nat := []
theorem row9391_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9391 row9391_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9391_block_013 : List Nat := []
theorem row9391_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9391 row9391_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9391_large_left_left : List Nat := row9391_small ++ row9391_block_005
theorem row9391_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9391 row9391_large_left_left :=
  Budget.RowUpperBound.branch row9391_small_covers row9391_block_005_covers

def row9391_large_left_right_right : List Nat := row9391_block_007 ++ row9391_block_008
theorem row9391_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9391 row9391_large_left_right_right :=
  Budget.RowUpperBound.branch row9391_block_007_covers row9391_block_008_covers

def row9391_large_left_right : List Nat := row9391_block_006 ++ row9391_large_left_right_right
theorem row9391_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9391 row9391_large_left_right :=
  Budget.RowUpperBound.branch row9391_block_006_covers row9391_large_left_right_right_covers

def row9391_large_left : List Nat := row9391_large_left_left ++ row9391_large_left_right
theorem row9391_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9391 row9391_large_left :=
  Budget.RowUpperBound.branch row9391_large_left_left_covers row9391_large_left_right_covers

def row9391_large_right_left : List Nat := row9391_block_009 ++ row9391_block_010
theorem row9391_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9391 row9391_large_right_left :=
  Budget.RowUpperBound.branch row9391_block_009_covers row9391_block_010_covers

def row9391_large_right_right_right : List Nat := row9391_block_012 ++ row9391_block_013
theorem row9391_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9391 row9391_large_right_right_right :=
  Budget.RowUpperBound.branch row9391_block_012_covers row9391_block_013_covers

def row9391_large_right_right : List Nat := row9391_block_011 ++ row9391_large_right_right_right
theorem row9391_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9391 row9391_large_right_right :=
  Budget.RowUpperBound.branch row9391_block_011_covers row9391_large_right_right_right_covers

def row9391_large_right : List Nat := row9391_large_right_left ++ row9391_large_right_right
theorem row9391_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9391 row9391_large_right :=
  Budget.RowUpperBound.branch row9391_large_right_left_covers row9391_large_right_right_covers

def row9391_large : List Nat := row9391_large_left ++ row9391_large_right
theorem row9391_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9391 row9391_large :=
  Budget.RowUpperBound.branch row9391_large_left_covers row9391_large_right_covers

def row9391 : List Nat := row9391_large
theorem row9391_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9391 row9391 := row9391_large_covers

def row9397_small : List Nat := []
theorem row9397_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9397 row9397_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9397_block_005 : List Nat := [9433]
def plan9397_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .second, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .second, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9397_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9397 row9397_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9397_block_005
  · decide
  · rfl

def row9397_block_006 : List Nat := [12409]
def plan9397_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 31), .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, .second, .first, (.secondFactor 23), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 11), .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 13), (.secondFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .second, .first, .second, (.secondFactor 7), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row9397_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9397 row9397_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9397_block_006
  · decide
  · rfl

def row9397_block_007 : List Nat := [13171, 13183, 13249]
def plan9397_block_007 : List Budget.Choice := [(.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 17), .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 23), (.secondFactor 17), (.firstFactor 13), .first, .second, (.firstFactor 19), (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .second, (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, (.firstFactor 13), .possible, .first, .first, (.secondFactor 11), .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 17), .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 31), (.firstFactor 19), .first, (.firstFactor 7), .first, .second, .second, .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9397_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9397 row9397_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9397_block_007
  · decide
  · rfl

def row9397_block_008 : List Nat := [14683, 14947, 15013, 15031, 16273]
def plan9397_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .possible, .first, .first, .first, .first, .possible, .first, .possible, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .second, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 17), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 37), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9397_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9397 row9397_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9397_block_008
  · decide
  · rfl

def row9397_block_009 : List Nat := []
def plan9397_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9397_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9397 row9397_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9397_block_009
  · decide
  · rfl

def row9397_block_010 : List Nat := []
theorem row9397_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9397 row9397_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9397_block_011 : List Nat := []
theorem row9397_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9397 row9397_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9397_block_012 : List Nat := []
theorem row9397_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9397 row9397_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9397_block_013 : List Nat := []
theorem row9397_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9397 row9397_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9397_large_left_left : List Nat := row9397_small ++ row9397_block_005
theorem row9397_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9397 row9397_large_left_left :=
  Budget.RowUpperBound.branch row9397_small_covers row9397_block_005_covers

def row9397_large_left_right_right : List Nat := row9397_block_007 ++ row9397_block_008
theorem row9397_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9397 row9397_large_left_right_right :=
  Budget.RowUpperBound.branch row9397_block_007_covers row9397_block_008_covers

def row9397_large_left_right : List Nat := row9397_block_006 ++ row9397_large_left_right_right
theorem row9397_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9397 row9397_large_left_right :=
  Budget.RowUpperBound.branch row9397_block_006_covers row9397_large_left_right_right_covers

def row9397_large_left : List Nat := row9397_large_left_left ++ row9397_large_left_right
theorem row9397_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9397 row9397_large_left :=
  Budget.RowUpperBound.branch row9397_large_left_left_covers row9397_large_left_right_covers

def row9397_large_right_left : List Nat := row9397_block_009 ++ row9397_block_010
theorem row9397_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9397 row9397_large_right_left :=
  Budget.RowUpperBound.branch row9397_block_009_covers row9397_block_010_covers

def row9397_large_right_right_right : List Nat := row9397_block_012 ++ row9397_block_013
theorem row9397_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9397 row9397_large_right_right_right :=
  Budget.RowUpperBound.branch row9397_block_012_covers row9397_block_013_covers

def row9397_large_right_right : List Nat := row9397_block_011 ++ row9397_large_right_right_right
theorem row9397_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9397 row9397_large_right_right :=
  Budget.RowUpperBound.branch row9397_block_011_covers row9397_large_right_right_right_covers

def row9397_large_right : List Nat := row9397_large_right_left ++ row9397_large_right_right
theorem row9397_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9397 row9397_large_right :=
  Budget.RowUpperBound.branch row9397_large_right_left_covers row9397_large_right_right_covers

def row9397_large : List Nat := row9397_large_left ++ row9397_large_right
theorem row9397_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9397 row9397_large :=
  Budget.RowUpperBound.branch row9397_large_left_covers row9397_large_right_covers

def row9397 : List Nat := row9397_large
theorem row9397_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9397 row9397 := row9397_large_covers

end ProjectEulerSolutions.PrimePairLarge
