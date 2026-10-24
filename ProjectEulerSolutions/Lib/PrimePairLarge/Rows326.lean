import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10723_small : List Nat := []
theorem row10723_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10723 row10723_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10723_block_005 : List Nat := []
theorem row10723_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10723 row10723_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10723_block_006 : List Nat := [10753, 11887, 12037]
def plan10723_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.secondFactor 37), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 11), .second, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .second, (.firstFactor 23), (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 37), .first, .first, .first, .second, (.firstFactor 31), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 37), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10723_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10723 row10723_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10723_block_006
  · decide
  · rfl

def row10723_block_007 : List Nat := [12517, 12697, 13003, 13381, 13873]
def plan10723_block_007 : List Budget.Choice := [(.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 31), (.secondFactor 23), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.secondFactor 31), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29)]
set_option maxRecDepth 2048 in
theorem row10723_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10723 row10723_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10723_block_007
  · decide
  · rfl

def row10723_block_008 : List Nat := [15289]
def plan10723_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .second, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10723_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10723 row10723_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10723_block_008
  · decide
  · rfl

def row10723_block_009 : List Nat := []
theorem row10723_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10723 row10723_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10723_block_010 : List Nat := []
theorem row10723_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10723 row10723_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10723_block_011 : List Nat := []
theorem row10723_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10723 row10723_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10723_block_012 : List Nat := []
theorem row10723_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10723 row10723_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10723_block_013 : List Nat := []
theorem row10723_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10723 row10723_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10723_large_left_left : List Nat := row10723_small ++ row10723_block_005
theorem row10723_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10723 row10723_large_left_left :=
  Budget.RowUpperBound.branch row10723_small_covers row10723_block_005_covers

def row10723_large_left_right_right : List Nat := row10723_block_007 ++ row10723_block_008
theorem row10723_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10723 row10723_large_left_right_right :=
  Budget.RowUpperBound.branch row10723_block_007_covers row10723_block_008_covers

def row10723_large_left_right : List Nat := row10723_block_006 ++ row10723_large_left_right_right
theorem row10723_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10723 row10723_large_left_right :=
  Budget.RowUpperBound.branch row10723_block_006_covers row10723_large_left_right_right_covers

def row10723_large_left : List Nat := row10723_large_left_left ++ row10723_large_left_right
theorem row10723_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10723 row10723_large_left :=
  Budget.RowUpperBound.branch row10723_large_left_left_covers row10723_large_left_right_covers

def row10723_large_right_left : List Nat := row10723_block_009 ++ row10723_block_010
theorem row10723_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10723 row10723_large_right_left :=
  Budget.RowUpperBound.branch row10723_block_009_covers row10723_block_010_covers

def row10723_large_right_right_right : List Nat := row10723_block_012 ++ row10723_block_013
theorem row10723_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10723 row10723_large_right_right_right :=
  Budget.RowUpperBound.branch row10723_block_012_covers row10723_block_013_covers

def row10723_large_right_right : List Nat := row10723_block_011 ++ row10723_large_right_right_right
theorem row10723_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10723 row10723_large_right_right :=
  Budget.RowUpperBound.branch row10723_block_011_covers row10723_large_right_right_right_covers

def row10723_large_right : List Nat := row10723_large_right_left ++ row10723_large_right_right
theorem row10723_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10723 row10723_large_right :=
  Budget.RowUpperBound.branch row10723_large_right_left_covers row10723_large_right_right_covers

def row10723_large : List Nat := row10723_large_left ++ row10723_large_right
theorem row10723_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10723 row10723_large :=
  Budget.RowUpperBound.branch row10723_large_left_covers row10723_large_right_covers

def row10723 : List Nat := row10723_large
theorem row10723_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10723 row10723 := row10723_large_covers

def row10729_small : List Nat := []
theorem row10729_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10729 row10729_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10729_block_005 : List Nat := []
theorem row10729_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10729 row10729_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10729_block_006 : List Nat := [10903, 11617, 11863]
def plan10729_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .possible, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .second, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 17), .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 37), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 13), .first, .possible, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .possible, .first, (.secondFactor 37), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .second, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row10729_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10729 row10729_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10729_block_006
  · decide
  · rfl

def row10729_block_007 : List Nat := [12853]
def plan10729_block_007 : List Budget.Choice := [(.firstFactor 7), .first, .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 37), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 17), (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 19), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 37), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 37), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10729_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10729 row10729_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10729_block_007
  · decide
  · rfl

def row10729_block_008 : List Nat := []
def plan10729_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 37), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .second, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10729_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10729 row10729_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10729_block_008
  · decide
  · rfl

def row10729_block_009 : List Nat := []
theorem row10729_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10729 row10729_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10729_block_010 : List Nat := []
theorem row10729_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10729 row10729_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10729_block_011 : List Nat := []
theorem row10729_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10729 row10729_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10729_block_012 : List Nat := []
theorem row10729_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10729 row10729_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10729_block_013 : List Nat := []
theorem row10729_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10729 row10729_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10729_large_left_left : List Nat := row10729_small ++ row10729_block_005
theorem row10729_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10729 row10729_large_left_left :=
  Budget.RowUpperBound.branch row10729_small_covers row10729_block_005_covers

def row10729_large_left_right_right : List Nat := row10729_block_007 ++ row10729_block_008
theorem row10729_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10729 row10729_large_left_right_right :=
  Budget.RowUpperBound.branch row10729_block_007_covers row10729_block_008_covers

def row10729_large_left_right : List Nat := row10729_block_006 ++ row10729_large_left_right_right
theorem row10729_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10729 row10729_large_left_right :=
  Budget.RowUpperBound.branch row10729_block_006_covers row10729_large_left_right_right_covers

def row10729_large_left : List Nat := row10729_large_left_left ++ row10729_large_left_right
theorem row10729_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10729 row10729_large_left :=
  Budget.RowUpperBound.branch row10729_large_left_left_covers row10729_large_left_right_covers

def row10729_large_right_left : List Nat := row10729_block_009 ++ row10729_block_010
theorem row10729_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10729 row10729_large_right_left :=
  Budget.RowUpperBound.branch row10729_block_009_covers row10729_block_010_covers

def row10729_large_right_right_right : List Nat := row10729_block_012 ++ row10729_block_013
theorem row10729_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10729 row10729_large_right_right_right :=
  Budget.RowUpperBound.branch row10729_block_012_covers row10729_block_013_covers

def row10729_large_right_right : List Nat := row10729_block_011 ++ row10729_large_right_right_right
theorem row10729_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10729 row10729_large_right_right :=
  Budget.RowUpperBound.branch row10729_block_011_covers row10729_large_right_right_right_covers

def row10729_large_right : List Nat := row10729_large_right_left ++ row10729_large_right_right
theorem row10729_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10729 row10729_large_right :=
  Budget.RowUpperBound.branch row10729_large_right_left_covers row10729_large_right_right_covers

def row10729_large : List Nat := row10729_large_left ++ row10729_large_right
theorem row10729_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10729 row10729_large :=
  Budget.RowUpperBound.branch row10729_large_left_covers row10729_large_right_covers

def row10729 : List Nat := row10729_large
theorem row10729_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10729 row10729 := row10729_large_covers

def row10733_small : List Nat := []
theorem row10733_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10733 row10733_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10733_block_005 : List Nat := []
theorem row10733_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10733 row10733_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10733_block_006 : List Nat := [10883, 11783, 11909]
def plan10733_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 13), .second, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 37), .first, .second, .first, .first, .first, .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .second, .possible, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 37), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row10733_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10733 row10733_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10733_block_006
  · decide
  · rfl

def row10733_block_007 : List Nat := [13313]
def plan10733_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 29), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 13), .second, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row10733_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10733 row10733_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10733_block_007
  · decide
  · rfl

def row10733_block_008 : List Nat := [14747]
def plan10733_block_008 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 37), (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 29), .first, .second, (.firstFactor 13), .first, .first, (.secondFactor 23), .possible, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10733_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10733 row10733_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10733_block_008
  · decide
  · rfl

def row10733_block_009 : List Nat := []
theorem row10733_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10733 row10733_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10733_block_010 : List Nat := []
theorem row10733_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10733 row10733_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10733_block_011 : List Nat := []
theorem row10733_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10733 row10733_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10733_block_012 : List Nat := []
theorem row10733_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10733 row10733_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10733_block_013 : List Nat := []
theorem row10733_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10733 row10733_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10733_large_left_left : List Nat := row10733_small ++ row10733_block_005
theorem row10733_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10733 row10733_large_left_left :=
  Budget.RowUpperBound.branch row10733_small_covers row10733_block_005_covers

def row10733_large_left_right_right : List Nat := row10733_block_007 ++ row10733_block_008
theorem row10733_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10733 row10733_large_left_right_right :=
  Budget.RowUpperBound.branch row10733_block_007_covers row10733_block_008_covers

def row10733_large_left_right : List Nat := row10733_block_006 ++ row10733_large_left_right_right
theorem row10733_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10733 row10733_large_left_right :=
  Budget.RowUpperBound.branch row10733_block_006_covers row10733_large_left_right_right_covers

def row10733_large_left : List Nat := row10733_large_left_left ++ row10733_large_left_right
theorem row10733_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10733 row10733_large_left :=
  Budget.RowUpperBound.branch row10733_large_left_left_covers row10733_large_left_right_covers

def row10733_large_right_left : List Nat := row10733_block_009 ++ row10733_block_010
theorem row10733_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10733 row10733_large_right_left :=
  Budget.RowUpperBound.branch row10733_block_009_covers row10733_block_010_covers

def row10733_large_right_right_right : List Nat := row10733_block_012 ++ row10733_block_013
theorem row10733_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10733 row10733_large_right_right_right :=
  Budget.RowUpperBound.branch row10733_block_012_covers row10733_block_013_covers

def row10733_large_right_right : List Nat := row10733_block_011 ++ row10733_large_right_right_right
theorem row10733_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10733 row10733_large_right_right :=
  Budget.RowUpperBound.branch row10733_block_011_covers row10733_large_right_right_right_covers

def row10733_large_right : List Nat := row10733_large_right_left ++ row10733_large_right_right
theorem row10733_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10733 row10733_large_right :=
  Budget.RowUpperBound.branch row10733_large_right_left_covers row10733_large_right_right_covers

def row10733_large : List Nat := row10733_large_left ++ row10733_large_right
theorem row10733_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10733 row10733_large :=
  Budget.RowUpperBound.branch row10733_large_left_covers row10733_large_right_covers

def row10733 : List Nat := row10733_large
theorem row10733_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10733 row10733 := row10733_large_covers

def row10739_small : List Nat := []
theorem row10739_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10739 row10739_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10739_block_005 : List Nat := []
theorem row10739_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10739 row10739_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10739_block_006 : List Nat := []
def plan10739_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 31), (.firstFactor 11), .second, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .second, .second, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10739_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10739 row10739_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10739_block_006
  · decide
  · rfl

def row10739_block_007 : List Nat := [12479, 12911, 13331, 14423]
def plan10739_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 17), .first, .possible, (.firstFactor 7), .first, .second, (.firstFactor 23), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 19), (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 37), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row10739_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10739 row10739_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10739_block_007
  · decide
  · rfl

def row10739_block_008 : List Nat := [14717, 14783]
def plan10739_block_008 : List Budget.Choice := [.first, .second, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 29), (.secondFactor 7), .second, .first, (.secondFactor 13), .first, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .second, .second, (.secondFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10739_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10739 row10739_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10739_block_008
  · decide
  · rfl

def row10739_block_009 : List Nat := []
theorem row10739_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10739 row10739_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10739_block_010 : List Nat := []
theorem row10739_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10739 row10739_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10739_block_011 : List Nat := []
theorem row10739_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10739 row10739_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10739_block_012 : List Nat := []
theorem row10739_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10739 row10739_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10739_block_013 : List Nat := []
theorem row10739_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10739 row10739_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10739_large_left_left : List Nat := row10739_small ++ row10739_block_005
theorem row10739_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10739 row10739_large_left_left :=
  Budget.RowUpperBound.branch row10739_small_covers row10739_block_005_covers

def row10739_large_left_right_right : List Nat := row10739_block_007 ++ row10739_block_008
theorem row10739_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10739 row10739_large_left_right_right :=
  Budget.RowUpperBound.branch row10739_block_007_covers row10739_block_008_covers

def row10739_large_left_right : List Nat := row10739_block_006 ++ row10739_large_left_right_right
theorem row10739_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10739 row10739_large_left_right :=
  Budget.RowUpperBound.branch row10739_block_006_covers row10739_large_left_right_right_covers

def row10739_large_left : List Nat := row10739_large_left_left ++ row10739_large_left_right
theorem row10739_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10739 row10739_large_left :=
  Budget.RowUpperBound.branch row10739_large_left_left_covers row10739_large_left_right_covers

def row10739_large_right_left : List Nat := row10739_block_009 ++ row10739_block_010
theorem row10739_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10739 row10739_large_right_left :=
  Budget.RowUpperBound.branch row10739_block_009_covers row10739_block_010_covers

def row10739_large_right_right_right : List Nat := row10739_block_012 ++ row10739_block_013
theorem row10739_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10739 row10739_large_right_right_right :=
  Budget.RowUpperBound.branch row10739_block_012_covers row10739_block_013_covers

def row10739_large_right_right : List Nat := row10739_block_011 ++ row10739_large_right_right_right
theorem row10739_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10739 row10739_large_right_right :=
  Budget.RowUpperBound.branch row10739_block_011_covers row10739_large_right_right_right_covers

def row10739_large_right : List Nat := row10739_large_right_left ++ row10739_large_right_right
theorem row10739_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10739 row10739_large_right :=
  Budget.RowUpperBound.branch row10739_large_right_left_covers row10739_large_right_right_covers

def row10739_large : List Nat := row10739_large_left ++ row10739_large_right
theorem row10739_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10739 row10739_large :=
  Budget.RowUpperBound.branch row10739_large_left_covers row10739_large_right_covers

def row10739 : List Nat := row10739_large
theorem row10739_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10739 row10739 := row10739_large_covers

end ProjectEulerSolutions.PrimePairLarge
