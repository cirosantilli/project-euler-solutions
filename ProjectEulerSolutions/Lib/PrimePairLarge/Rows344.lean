import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11437_small : List Nat := []
theorem row11437_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11437 row11437_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11437_block_005 : List Nat := []
theorem row11437_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11437 row11437_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11437_block_006 : List Nat := [11971, 12301]
def plan11437_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 13), (.firstFactor 19), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 11), .first, .second, (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .second, (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11437_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11437 row11437_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11437_block_006
  · decide
  · rfl

def row11437_block_007 : List Nat := [13729, 13789, 14251, 14323, 14431]
def plan11437_block_007 : List Budget.Choice := [(.firstFactor 23), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), (.firstFactor 31), (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 37), .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 29), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 19), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 37), .first, .first, .second]
set_option maxRecDepth 2048 in
theorem row11437_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11437 row11437_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11437_block_007
  · decide
  · rfl

def row11437_block_008 : List Nat := []
def plan11437_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11437_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11437 row11437_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11437_block_008
  · decide
  · rfl

def row11437_block_009 : List Nat := []
theorem row11437_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11437 row11437_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11437_block_010 : List Nat := []
theorem row11437_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11437 row11437_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11437_block_011 : List Nat := []
theorem row11437_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11437 row11437_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11437_block_012 : List Nat := []
theorem row11437_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11437 row11437_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11437_block_013 : List Nat := []
theorem row11437_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11437 row11437_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11437_large_left_left : List Nat := row11437_small ++ row11437_block_005
theorem row11437_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11437 row11437_large_left_left :=
  Budget.RowUpperBound.branch row11437_small_covers row11437_block_005_covers

def row11437_large_left_right_right : List Nat := row11437_block_007 ++ row11437_block_008
theorem row11437_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11437 row11437_large_left_right_right :=
  Budget.RowUpperBound.branch row11437_block_007_covers row11437_block_008_covers

def row11437_large_left_right : List Nat := row11437_block_006 ++ row11437_large_left_right_right
theorem row11437_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11437 row11437_large_left_right :=
  Budget.RowUpperBound.branch row11437_block_006_covers row11437_large_left_right_right_covers

def row11437_large_left : List Nat := row11437_large_left_left ++ row11437_large_left_right
theorem row11437_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11437 row11437_large_left :=
  Budget.RowUpperBound.branch row11437_large_left_left_covers row11437_large_left_right_covers

def row11437_large_right_left : List Nat := row11437_block_009 ++ row11437_block_010
theorem row11437_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11437 row11437_large_right_left :=
  Budget.RowUpperBound.branch row11437_block_009_covers row11437_block_010_covers

def row11437_large_right_right_right : List Nat := row11437_block_012 ++ row11437_block_013
theorem row11437_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11437 row11437_large_right_right_right :=
  Budget.RowUpperBound.branch row11437_block_012_covers row11437_block_013_covers

def row11437_large_right_right : List Nat := row11437_block_011 ++ row11437_large_right_right_right
theorem row11437_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11437 row11437_large_right_right :=
  Budget.RowUpperBound.branch row11437_block_011_covers row11437_large_right_right_right_covers

def row11437_large_right : List Nat := row11437_large_right_left ++ row11437_large_right_right
theorem row11437_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11437 row11437_large_right :=
  Budget.RowUpperBound.branch row11437_large_right_left_covers row11437_large_right_right_covers

def row11437_large : List Nat := row11437_large_left ++ row11437_large_right
theorem row11437_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11437 row11437_large :=
  Budget.RowUpperBound.branch row11437_large_left_covers row11437_large_right_covers

def row11437 : List Nat := row11437_large
theorem row11437_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11437 row11437 := row11437_large_covers

def row11443_small : List Nat := []
theorem row11443_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11443 row11443_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11443_block_005 : List Nat := []
theorem row11443_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11443 row11443_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11443_block_006 : List Nat := []
def plan11443_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .second, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 7), .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11443_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11443 row11443_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11443_block_006
  · decide
  · rfl

def row11443_block_007 : List Nat := [12553, 14011, 14341, 14449]
def plan11443_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 7), .possible, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 19), (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 31), .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .second, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row11443_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11443 row11443_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11443_block_007
  · decide
  · rfl

def row11443_block_008 : List Nat := []
def plan11443_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 37), .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11443_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11443 row11443_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11443_block_008
  · decide
  · rfl

def row11443_block_009 : List Nat := []
theorem row11443_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11443 row11443_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11443_block_010 : List Nat := []
theorem row11443_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11443 row11443_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11443_block_011 : List Nat := []
theorem row11443_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11443 row11443_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11443_block_012 : List Nat := []
theorem row11443_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11443 row11443_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11443_block_013 : List Nat := []
theorem row11443_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11443 row11443_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11443_large_left_left : List Nat := row11443_small ++ row11443_block_005
theorem row11443_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11443 row11443_large_left_left :=
  Budget.RowUpperBound.branch row11443_small_covers row11443_block_005_covers

def row11443_large_left_right_right : List Nat := row11443_block_007 ++ row11443_block_008
theorem row11443_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11443 row11443_large_left_right_right :=
  Budget.RowUpperBound.branch row11443_block_007_covers row11443_block_008_covers

def row11443_large_left_right : List Nat := row11443_block_006 ++ row11443_large_left_right_right
theorem row11443_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11443 row11443_large_left_right :=
  Budget.RowUpperBound.branch row11443_block_006_covers row11443_large_left_right_right_covers

def row11443_large_left : List Nat := row11443_large_left_left ++ row11443_large_left_right
theorem row11443_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11443 row11443_large_left :=
  Budget.RowUpperBound.branch row11443_large_left_left_covers row11443_large_left_right_covers

def row11443_large_right_left : List Nat := row11443_block_009 ++ row11443_block_010
theorem row11443_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11443 row11443_large_right_left :=
  Budget.RowUpperBound.branch row11443_block_009_covers row11443_block_010_covers

def row11443_large_right_right_right : List Nat := row11443_block_012 ++ row11443_block_013
theorem row11443_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11443 row11443_large_right_right_right :=
  Budget.RowUpperBound.branch row11443_block_012_covers row11443_block_013_covers

def row11443_large_right_right : List Nat := row11443_block_011 ++ row11443_large_right_right_right
theorem row11443_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11443 row11443_large_right_right :=
  Budget.RowUpperBound.branch row11443_block_011_covers row11443_large_right_right_right_covers

def row11443_large_right : List Nat := row11443_large_right_left ++ row11443_large_right_right
theorem row11443_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11443 row11443_large_right :=
  Budget.RowUpperBound.branch row11443_large_right_left_covers row11443_large_right_right_covers

def row11443_large : List Nat := row11443_large_left ++ row11443_large_right
theorem row11443_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11443 row11443_large :=
  Budget.RowUpperBound.branch row11443_large_left_covers row11443_large_right_covers

def row11443 : List Nat := row11443_large
theorem row11443_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11443 row11443 := row11443_large_covers

def row11447_small : List Nat := []
theorem row11447_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11447 row11447_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11447_block_005 : List Nat := []
theorem row11447_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11447 row11447_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11447_block_006 : List Nat := [11483, 11783, 11927, 12011, 12071, 12203]
def plan11447_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .second, .first, .first, .first, .second, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .possible, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .possible, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 31), (.firstFactor 7), .first, .second, .first, .first, .possible, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 31), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row11447_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11447 row11447_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11447_block_006
  · decide
  · rfl

def row11447_block_007 : List Nat := [13709, 13859, 14369]
def plan11447_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 13), .first, (.secondFactor 7), .second, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .second, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .possible, .first, .second, .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, (.secondFactor 29), .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11447_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11447 row11447_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11447_block_007
  · decide
  · rfl

def row11447_block_008 : List Nat := []
def plan11447_block_008 : List Budget.Choice := [.first, (.firstFactor 19), .first, (.firstFactor 13), .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11447_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11447 row11447_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11447_block_008
  · decide
  · rfl

def row11447_block_009 : List Nat := []
theorem row11447_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11447 row11447_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11447_block_010 : List Nat := []
theorem row11447_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11447 row11447_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11447_block_011 : List Nat := []
theorem row11447_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11447 row11447_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11447_block_012 : List Nat := []
theorem row11447_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11447 row11447_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11447_block_013 : List Nat := []
theorem row11447_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11447 row11447_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11447_large_left_left : List Nat := row11447_small ++ row11447_block_005
theorem row11447_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11447 row11447_large_left_left :=
  Budget.RowUpperBound.branch row11447_small_covers row11447_block_005_covers

def row11447_large_left_right_right : List Nat := row11447_block_007 ++ row11447_block_008
theorem row11447_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11447 row11447_large_left_right_right :=
  Budget.RowUpperBound.branch row11447_block_007_covers row11447_block_008_covers

def row11447_large_left_right : List Nat := row11447_block_006 ++ row11447_large_left_right_right
theorem row11447_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11447 row11447_large_left_right :=
  Budget.RowUpperBound.branch row11447_block_006_covers row11447_large_left_right_right_covers

def row11447_large_left : List Nat := row11447_large_left_left ++ row11447_large_left_right
theorem row11447_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11447 row11447_large_left :=
  Budget.RowUpperBound.branch row11447_large_left_left_covers row11447_large_left_right_covers

def row11447_large_right_left : List Nat := row11447_block_009 ++ row11447_block_010
theorem row11447_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11447 row11447_large_right_left :=
  Budget.RowUpperBound.branch row11447_block_009_covers row11447_block_010_covers

def row11447_large_right_right_right : List Nat := row11447_block_012 ++ row11447_block_013
theorem row11447_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11447 row11447_large_right_right_right :=
  Budget.RowUpperBound.branch row11447_block_012_covers row11447_block_013_covers

def row11447_large_right_right : List Nat := row11447_block_011 ++ row11447_large_right_right_right
theorem row11447_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11447 row11447_large_right_right :=
  Budget.RowUpperBound.branch row11447_block_011_covers row11447_large_right_right_right_covers

def row11447_large_right : List Nat := row11447_large_right_left ++ row11447_large_right_right
theorem row11447_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11447 row11447_large_right :=
  Budget.RowUpperBound.branch row11447_large_right_left_covers row11447_large_right_right_covers

def row11447_large : List Nat := row11447_large_left ++ row11447_large_right
theorem row11447_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11447 row11447_large :=
  Budget.RowUpperBound.branch row11447_large_left_covers row11447_large_right_covers

def row11447 : List Nat := row11447_large
theorem row11447_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11447 row11447 := row11447_large_covers

def row11467_small : List Nat := []
theorem row11467_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11467 row11467_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11467_block_005 : List Nat := []
theorem row11467_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11467 row11467_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11467_block_006 : List Nat := [11887]
def plan11467_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11467_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11467 row11467_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11467_block_006
  · decide
  · rfl

def row11467_block_007 : List Nat := [12421, 12487, 12979]
def plan11467_block_007 : List Budget.Choice := [.possible, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, .first, (.secondFactor 17), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 29), (.secondFactor 19), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 31), (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 37), .first, .first, (.secondFactor 19), .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, (.secondFactor 29), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 29), .second, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .second, (.secondFactor 31)]
set_option maxRecDepth 2048 in
theorem row11467_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11467 row11467_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11467_block_007
  · decide
  · rfl

def row11467_block_008 : List Nat := []
def plan11467_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11467_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11467 row11467_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11467_block_008
  · decide
  · rfl

def row11467_block_009 : List Nat := []
theorem row11467_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11467 row11467_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11467_block_010 : List Nat := []
theorem row11467_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11467 row11467_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11467_block_011 : List Nat := []
theorem row11467_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11467 row11467_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11467_block_012 : List Nat := []
theorem row11467_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11467 row11467_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11467_block_013 : List Nat := []
theorem row11467_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11467 row11467_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11467_large_left_left : List Nat := row11467_small ++ row11467_block_005
theorem row11467_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11467 row11467_large_left_left :=
  Budget.RowUpperBound.branch row11467_small_covers row11467_block_005_covers

def row11467_large_left_right_right : List Nat := row11467_block_007 ++ row11467_block_008
theorem row11467_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11467 row11467_large_left_right_right :=
  Budget.RowUpperBound.branch row11467_block_007_covers row11467_block_008_covers

def row11467_large_left_right : List Nat := row11467_block_006 ++ row11467_large_left_right_right
theorem row11467_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11467 row11467_large_left_right :=
  Budget.RowUpperBound.branch row11467_block_006_covers row11467_large_left_right_right_covers

def row11467_large_left : List Nat := row11467_large_left_left ++ row11467_large_left_right
theorem row11467_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11467 row11467_large_left :=
  Budget.RowUpperBound.branch row11467_large_left_left_covers row11467_large_left_right_covers

def row11467_large_right_left : List Nat := row11467_block_009 ++ row11467_block_010
theorem row11467_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11467 row11467_large_right_left :=
  Budget.RowUpperBound.branch row11467_block_009_covers row11467_block_010_covers

def row11467_large_right_right_right : List Nat := row11467_block_012 ++ row11467_block_013
theorem row11467_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11467 row11467_large_right_right_right :=
  Budget.RowUpperBound.branch row11467_block_012_covers row11467_block_013_covers

def row11467_large_right_right : List Nat := row11467_block_011 ++ row11467_large_right_right_right
theorem row11467_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11467 row11467_large_right_right :=
  Budget.RowUpperBound.branch row11467_block_011_covers row11467_large_right_right_right_covers

def row11467_large_right : List Nat := row11467_large_right_left ++ row11467_large_right_right
theorem row11467_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11467 row11467_large_right :=
  Budget.RowUpperBound.branch row11467_large_right_left_covers row11467_large_right_right_covers

def row11467_large : List Nat := row11467_large_left ++ row11467_large_right
theorem row11467_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11467 row11467_large :=
  Budget.RowUpperBound.branch row11467_large_left_covers row11467_large_right_covers

def row11467 : List Nat := row11467_large
theorem row11467_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11467 row11467 := row11467_large_covers

end ProjectEulerSolutions.PrimePairLarge
