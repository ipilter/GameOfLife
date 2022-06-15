#include <gl/glew.h> // must be before any OpenGL related header

#include "Mesh.h"

Mesh::Mesh( const std::vector<float>& points
            , const std::vector<uint32_t>& indices
            , const uint32_t stride
            , const uint32_t vertexOffset
            , const uint32_t worldTexelOffset
            , const size_t patternTexelOffset )
  : mIndexCount( static_cast<uint32_t>( indices.size() ) )
{
  glGenBuffers( 1, &mVbo );
  glBindBuffer( GL_ARRAY_BUFFER, mVbo );
  glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * points.size(), &points.front(), GL_STATIC_DRAW );

  glGenBuffers( 1, &mIbo );
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mIbo );
  glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( uint32_t ) * indices.size(), &indices.front(), GL_STATIC_DRAW );

  glGenVertexArrays( 1, &mVao );
  glBindVertexArray( mVao );
  glEnableVertexAttribArray( 0 );
  glEnableVertexAttribArray( 1 );
  glEnableVertexAttribArray( 2 );
  glBindBuffer( GL_ARRAY_BUFFER, mVbo );
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mIbo );

  glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>( static_cast<size_t>( vertexOffset ) ) );
  glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>( static_cast<size_t>( worldTexelOffset ) ) );
  glVertexAttribPointer( 2, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>( static_cast<size_t>( patternTexelOffset ) ) );
  glBindVertexArray( 0 );
}

Mesh::~Mesh()
{
  glDeleteVertexArrays( 1, &mVao );
  glDeleteBuffers( 1, &mVbo );
  glDeleteBuffers( 1, &mIbo );
}

void Mesh::Bind()
{
  glBindVertexArray( mVao );
}

void Mesh::Unbind()
{
  glBindVertexArray( 0 );
}

void Mesh::Render()
{
  glDrawElements( GL_TRIANGLES, mIndexCount, GL_UNSIGNED_INT, 0 );
}
