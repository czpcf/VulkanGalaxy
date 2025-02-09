#include <vulkan/vulkan.h>

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <iostream>
#include <array>
#include <vector>
#include <set>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <optional>
#include <random>
#include "utils.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
const uint32_t WINDOW_WIDTH = 800;
const uint32_t WINDOW_HEIGHT = 600;
const std::string APP_NAME = "VulkanGalaxy";
const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const float SPEED_SCALE = 1.0f;

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilyIndices {
	std::optional<uint32_t> uniformFamily; // uniform = graphics + compute
	std::optional<uint32_t> presentFamily;

	bool isComplete() const {
		return uniformFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapchainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct MVP {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
	alignas(64) float deltaTime; // TODO: add alignment check for uniform buffer(change sizoeof(MVP) into some function)
};

struct Vertex { // TODO: change alignment to save memory
	alignas(16) glm::vec3 pos;
	alignas(16) glm::vec3 color;
	alignas(8)  glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
		return attributeDescriptions;
	}
};

static std::vector<const char*> getRequiredExtensions();
static bool isDeviceSuitable(VkPhysicalDevice, VkSurfaceKHR&);
static QueueFamilyIndices findQueueFamilyProperties(VkPhysicalDevice, VkSurfaceKHR&);
static SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice, VkSurfaceKHR&);
static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>&);
static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>&);
static VkExtent2D chooseSwapExtent(GLFWwindow*, const VkSurfaceCapabilitiesKHR&);
static VkImageView createImageView(VkDevice, VkImage, VkFormat, VkImageAspectFlags, uint32_t);
static uint32_t findMemoryType(VkPhysicalDevice, uint32_t, VkMemoryPropertyFlags);
static VkFormat findSupportedFormat(VkPhysicalDevice, const std::vector<VkFormat>&, VkImageTiling, VkFormatFeatureFlags);
static bool hasStencilComponent(VkFormat);
static VkFormat findDepthFormat(VkPhysicalDevice);
static void createBuffer(VkPhysicalDevice, VkDevice, VkDeviceSize, VkBufferUsageFlags, VkMemoryPropertyFlags, VkBuffer&, VkDeviceMemory&);
static VkCommandBuffer beginSingleTimeCommands(VkDevice, VkCommandPool);
static void endSingleTimeCommands(VkDevice, VkQueue, VkCommandPool, VkCommandBuffer);
static void copyBufferToImage(VkDevice, VkQueue, VkCommandPool, VkBuffer, VkImage, uint32_t, uint32_t);
static void generateMipmaps(VkPhysicalDevice, VkDevice, VkQueue, VkCommandPool, VkImage, VkFormat, int32_t, int32_t, uint32_t);
static void transitionImageLayout(VkDevice, VkQueue, VkCommandPool, VkImage, VkFormat, VkImageLayout, VkImageLayout, uint32_t);
static void createImage(VkPhysicalDevice, VkDevice, uint32_t, uint32_t, uint32_t, VkFormat, VkSampleCountFlagBits, VkImageTiling, VkImageUsageFlags, VkMemoryPropertyFlags, VkImage&, VkDeviceMemory&);
static void copyBuffer(VkDevice, VkQueue, VkCommandPool, VkBuffer, VkBuffer, VkDeviceSize);
static std::vector<char> readFile(const std::string&);
static VkShaderModule createShaderModule(VkDevice, const std::vector<char>&);

class Model {
public:
	VkBuffer vertexBuffer;
	VkBuffer indexBuffer;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	float sizeScale;
	float speed;
	float speedSelf;
	glm::vec3 origin;
	glm::vec3 pivot;

	Model() {}
	Model(std::string _modelPath, std::string _texturePath, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, VkCommandPool commandPool,
		float scale = 1.0f, float _speed = 1.0f, float _speedSelf = 1.0f, glm::vec3 _origin = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 _pivot = glm::vec3(0.0f, 0.0f, 0.0f)) {
		createModel(_modelPath, _texturePath, physicalDevice, device, queue, commandPool, scale, _speed, _speedSelf, _origin, _pivot);
	}

	void createModel(std::string _modelPath, std::string _texturePath, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, VkCommandPool commandPool,
		float scale = 1.0f, float _speed = 1.0f, float _speedSelf = 1.0f, glm::vec3 _origin = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 _pivot = glm::vec3(0.0f, 0.0f, 0.0f)) {
		modelPath = _modelPath;
		texturePath = _texturePath;
		createTextureImage(physicalDevice, device, queue, commandPool);
		createTextureImageView(device);
		createTextureSampler(physicalDevice, device);
		loadModel();
		createVertexBuffer(physicalDevice, device, queue, commandPool);
		createIndexBuffer(physicalDevice, device, queue, commandPool);
		sizeScale = scale;
		speed = _speed;
		speedSelf = _speedSelf;
		origin = _origin;
		pivot = _pivot;
	}

	void destroy(VkDevice device) const {
		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);
		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);
		vkDestroyImage(device, textureImage, nullptr);
		vkDestroyImageView(device, textureImageView, nullptr);
		vkDestroySampler(device, textureSampler, nullptr);
		vkFreeMemory(device, textureImageMemory, nullptr);
	}

	VkDescriptorImageInfo getDescriptorImageInfo() {
		return VkDescriptorImageInfo{
			.sampler = textureSampler,
			.imageView = textureImageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};
	}
private:
	uint32_t mipLevels;
	VkImage textureImage;
	VkImageView textureImageView;
	VkDeviceMemory textureImageMemory;
	VkSampler textureSampler;
	std::string modelPath;
	std::string texturePath;
	VkDeviceMemory vertexBufferMemory;
	VkDeviceMemory indexBufferMemory;

	// get texture
	void createTextureImage(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, VkCommandPool commandPool) {
		int width, height, channels;
		stbi_uc* pixels = stbi_load(texturePath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
		VkDeviceSize imageSize = width * height * 4;
		if (!pixels) {
			throw std::runtime_error("failed to load texture");
		}
		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(physicalDevice, device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);
		stbi_image_free(pixels);

		createImage(physicalDevice, device, width, height, mipLevels, VK_FORMAT_R8G8B8A8_SRGB, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

		transitionImageLayout(device, queue, commandPool, textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
		copyBufferToImage(device, queue, commandPool, stagingBuffer, textureImage, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
		generateMipmaps(physicalDevice, device, queue, commandPool, textureImage, VK_FORMAT_R8G8B8A8_SRGB, width, height, mipLevels);
	}

	void createTextureImageView(VkDevice device) {
		textureImageView = createImageView(device, textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
	}

	void createTextureSampler(VkPhysicalDevice physicalDevice, VkDevice device) {
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);
		VkSamplerCreateInfo samplerInfo{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy = properties.limits.maxSamplerAnisotropy,
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.,
			.maxLod = static_cast<float>(mipLevels),
			.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		UT_CHECK_ERR(vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) == VK_SUCCESS, "failed to create texture sampler");
	}

	void loadModel() {
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str())) {
			throw std::runtime_error(warn + err);
		}

		for (const auto& shape : shapes) {
			for (const auto& index : shape.mesh.indices) {
				Vertex vertex{};
				vertex.pos = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				vertex.texCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};


				vertex.color = { 1.0f, 1.0f, 1.0f };
				vertices.push_back(vertex);
				indices.push_back(static_cast<uint32_t>(indices.size()));
			}
		}

	}

	void createVertexBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, VkCommandPool commandPool) {
		VkDeviceSize size = sizeof(vertices[0]) * vertices.size();
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(physicalDevice, device, size,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer, stagingBufferMemory);
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, size, 0, &data);
		memcpy(data, vertices.data(), (size_t)size);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(physicalDevice, device, size,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			vertexBuffer, vertexBufferMemory);
		copyBuffer(device, queue, commandPool, stagingBuffer, vertexBuffer, size);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createIndexBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, VkCommandPool commandPool) {
		VkDeviceSize size = sizeof(indices[0]) * indices.size();
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(physicalDevice, device, size,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer, stagingBufferMemory);
		void* data;
		UT_CHECK_ERR(vkMapMemory(device, stagingBufferMemory, 0, size, 0, &data) == VK_SUCCESS, "failed to map memory");
		memcpy(data, indices.data(), (size_t)size);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(physicalDevice, device, size,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			indexBuffer, indexBufferMemory);
		copyBuffer(device, queue, commandPool, stagingBuffer, indexBuffer, size);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
};

class Image2D {
public:
	// create a 2D image and bind it to the memory
	void create(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format,
		VkSampleCountFlagBits numSamples, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImageAspectFlags aspectFlags) {
		VkImageCreateInfo imageInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.flags = 0,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = format,
			.extent = {
				.width = width,
				.height = height,
				.depth = 1,
			},
			.mipLevels = mipLevels,
			.arrayLayers = 1,
			.samples = numSamples,
			.tiling = tiling,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		UT_CHECK_ERR(vkCreateImage(device, &imageInfo, nullptr, &image) == VK_SUCCESS, "failed to create image");
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);
		VkMemoryAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties),
		};
		UT_CHECK_ERR(vkAllocateMemory(device, &allocInfo, nullptr, &memory) == VK_SUCCESS, "failed to allocate image momory");
		UT_CHECK_ERR(vkBindImageMemory(device, image, memory, 0) == VK_SUCCESS, "failed to bind image to memory");
		imageView = createImageView(device, image, format, aspectFlags, mipLevels);
	}
	void destroy(VkDevice device) const {
		vkDestroyImageView(device, imageView, nullptr);
		vkDestroyImage(device, image, nullptr);
		vkFreeMemory(device, memory, nullptr);
	}
	VkImageView imageView;
private:
	VkImage image;
	VkDeviceMemory memory;
};

struct ParticleObject {
	alignas(16) glm::vec3 x;
	alignas(16) glm::vec3 colorX;
	alignas(8)  glm::vec2 texCoordX;

	alignas(16) glm::vec3 y;
	alignas(16) glm::vec3 colorY;
	alignas(8)  glm::vec2 texCoordY;

	alignas(16) glm::vec3 z;
	alignas(16) glm::vec3 colorZ;
	alignas(8)  glm::vec2 texCoordZ;

	alignas(16) glm::vec3 w;
	alignas(16) glm::vec3 colorW;
	alignas(8)  glm::vec2 texCoordW;

	alignas(4) float radius;
	alignas(4) float theta;
};


struct ParticleObjectOutput {
	alignas(16) glm::vec3 worldX;
	alignas(16) glm::vec3 colorX;
	alignas(8)  glm::vec2 texCoordX;

	alignas(16) glm::vec3 worldY;
	alignas(16) glm::vec3 colorY;
	alignas(8)  glm::vec2 texCoordY;

	alignas(16) glm::vec3 worldZ;
	alignas(16) glm::vec3 colorZ;
	alignas(8)  glm::vec2 texCoordZ;

	alignas(16) glm::vec3 worldW;
	alignas(16) glm::vec3 colorW;
	alignas(8)  glm::vec2 texCoordW;
};


class ParticleSystem {
public:
	std::vector<VkBuffer> objectBuffer;
	std::vector<VkBuffer> vertexBuffer;
	VkBuffer indexBuffer;
	std::vector<ParticleObject> particles;
	std::vector<uint32_t> indices;

	static std::array<glm::vec3, 4> makeTetrahedronVertices(float size) {
		std::array<glm::vec3, 4> vertices = {
			glm::vec3(0.0f, 0.0f, 1.0f),
			glm::vec3(2.0f * sqrt(2.0f) / 3.0f, 0.0f, -1.0f / 3.0f),
			glm::vec3(-sqrt(2.0f) / 3.0f, sqrt(6.0f) / 3.0f, -1.0f / 3.0f),
			glm::vec3(-sqrt(2.0f) / 3.0f, -sqrt(6.0f) / 3.0f, -1.0f / 3.0f)
		};

		for (auto& vertex : vertices) {
			vertex = glm::normalize(vertex) * size;
		}

		return vertices;
	}

	static glm::mat3 makeRandomRotation(std::default_random_engine& rndEngine) {
		std::uniform_real_distribution<float> rndAngle(0.0f, 2.0f * 3.14159265358979323846f);
		float angleX = rndAngle(rndEngine);
		float angleY = rndAngle(rndEngine);
		float angleZ = rndAngle(rndEngine);

		glm::mat3 rotationX = glm::mat3(1.0f, 0.0f, 0.0f,
			0.0f, cos(angleX), -sin(angleX),
			0.0f, sin(angleX), cos(angleX));

		glm::mat3 rotationY = glm::mat3(cos(angleY), 0.0f, sin(angleY),
			0.0f, 1.0f, 0.0f,
			-sin(angleY), 0.0f, cos(angleY));

		glm::mat3 rotationZ = glm::mat3(cos(angleZ), -sin(angleZ), 0.0f,
			sin(angleZ), cos(angleZ), 0.0f,
			0.0f, 0.0f, 1.0f);

		return rotationZ * rotationY * rotationX;
	}


	void create(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, VkCommandPool commandPool, uint32_t particleNumbers, std::string _texturePath) {
		n = particleNumbers;
		particles.resize(n);
		texturePath = _texturePath;

		std::default_random_engine rndEngine(123);
		const float MIN_RADIUS = 4.0f;
		const float MAX_RADIUS = 20.0f;
		std::uniform_real_distribution<float> rndDist(MIN_RADIUS, MAX_RADIUS);
		std::uniform_real_distribution<float> rndUniform(0.0f, 1.0f);

		for (int i = 0; i < n; ++i) {
			float size = (rndUniform(rndEngine) + 0.5) * 0.04;
			float radius = rndDist(rndEngine);
			radius = (radius - MIN_RADIUS) / (MAX_RADIUS - MIN_RADIUS);
			radius = radius * radius + MIN_RADIUS;
			float theta = rndUniform(rndEngine) * 2.0f * 3.14159265358979323846f;

			auto vertices = makeTetrahedronVertices(size);

			glm::mat3 rotation = makeRandomRotation(rndEngine);

			particles[i].x = rotation * vertices[0];
			particles[i].colorX = glm::vec3(1.0f, 1.0f, 1.0f);
			particles[i].y = rotation * vertices[1];
			particles[i].colorY = glm::vec3(1.0f, 1.0f, 1.0f);
			particles[i].z = rotation * vertices[2];
			particles[i].colorZ = glm::vec3(1.0f, 1.0f, 1.0f);
			particles[i].w = rotation * vertices[3];
			particles[i].colorW = glm::vec3(1.0f, 1.0f, 1.0f);
			particles[i].radius = radius;
			particles[i].theta = theta;
		}
		createObjectBuffer(physicalDevice, device, queue, commandPool);
		createIndexBuffer(physicalDevice, device, queue, commandPool);
		createTextureImage(physicalDevice, device, queue, commandPool);
		createTextureImageView(device);
		createTextureSampler(physicalDevice, device);
	}

	void destroy(VkDevice device) const {
		for (int i = 0; i < objectBuffer.size(); ++i) {
			vkDestroyBuffer(device, objectBuffer[i], nullptr);
			vkFreeMemory(device, objectBufferMemory[i], nullptr);
			vkDestroyBuffer(device, vertexBuffer[i], nullptr);
			vkFreeMemory(device, vertexBufferMemory[i], nullptr);
		}
		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);
		vkDestroyImage(device, textureImage, nullptr);
		vkDestroyImageView(device, textureImageView, nullptr);
		vkDestroySampler(device, textureSampler, nullptr);
		vkFreeMemory(device, textureImageMemory, nullptr);
	}

	VkDescriptorImageInfo getDescriptorImageInfo() {
		return VkDescriptorImageInfo{
			.sampler = textureSampler,
			.imageView = textureImageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};
	}

	VkDescriptorBufferInfo getDescriptorObjectBufferInfo(uint32_t index) {
		return VkDescriptorBufferInfo{
			.buffer = objectBuffer[index],
			.offset = 0,
			.range = sizeof(ParticleObject) * particles.size(),
		};
	}

	VkDescriptorBufferInfo getDescriptorVertexBufferInfo(uint32_t index) {
		return VkDescriptorBufferInfo{
			.buffer = vertexBuffer[index],
			.offset = 0,
			.range = sizeof(ParticleObjectOutput) * particles.size(),
		};
	}
private:
	uint32_t n;
	std::vector<VkDeviceMemory> objectBufferMemory;
	std::vector<VkDeviceMemory> vertexBufferMemory;
	VkDeviceMemory indexBufferMemory;

	uint32_t mipLevels;
	VkImage textureImage;
	VkImageView textureImageView;
	VkDeviceMemory textureImageMemory;
	VkSampler textureSampler;
	std::string texturePath;

	// get texture
	void createTextureImage(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, VkCommandPool commandPool) {
		int width, height, channels;
		stbi_uc* pixels = stbi_load(texturePath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
		VkDeviceSize imageSize = width * height * 4;
		if (!pixels) {
			throw std::runtime_error("failed to load texture");
		}
		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(physicalDevice, device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);
		stbi_image_free(pixels);

		createImage(physicalDevice, device, width, height, mipLevels, VK_FORMAT_R8G8B8A8_SRGB, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

		transitionImageLayout(device, queue, commandPool, textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
		copyBufferToImage(device, queue, commandPool, stagingBuffer, textureImage, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
		generateMipmaps(physicalDevice, device, queue, commandPool, textureImage, VK_FORMAT_R8G8B8A8_SRGB, width, height, mipLevels);
	}

	void createTextureImageView(VkDevice device) {
		textureImageView = createImageView(device, textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
	}

	void createTextureSampler(VkPhysicalDevice physicalDevice, VkDevice device) {
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);
		VkSamplerCreateInfo samplerInfo{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy = properties.limits.maxSamplerAnisotropy,
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.,
			.maxLod = static_cast<float>(mipLevels),
			.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		UT_CHECK_ERR(vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) == VK_SUCCESS, "failed to create texture sampler");
	}

	void createObjectBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, VkCommandPool commandPool) {
		VkDeviceSize sizeObject = sizeof(particles[0]) * particles.size();
		VkDeviceSize sizeVertex = sizeof(ParticleObjectOutput) * particles.size();
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(physicalDevice, device, sizeObject,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer, stagingBufferMemory);
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, sizeObject, 0, &data);
		memcpy(data, particles.data(), (size_t)sizeObject);
		vkUnmapMemory(device, stagingBufferMemory);

		objectBuffer.resize(MAX_FRAMES_IN_FLIGHT);
		objectBufferMemory.resize(MAX_FRAMES_IN_FLIGHT);
		vertexBuffer.resize(MAX_FRAMES_IN_FLIGHT);
		vertexBufferMemory.resize(MAX_FRAMES_IN_FLIGHT);
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			createBuffer(physicalDevice, device, sizeObject,
				VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				objectBuffer[i], objectBufferMemory[i]);
			copyBuffer(device, queue, commandPool, stagingBuffer, objectBuffer[i], sizeObject);
			createBuffer(physicalDevice, device, sizeVertex,
				VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				vertexBuffer[i], vertexBufferMemory[i]);
		}
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createIndexBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, VkCommandPool commandPool) {
		indices.clear();
		uint32_t bias = 0;
		for (int i = 0; i < particles.size(); ++i) {
			indices.push_back(bias + 0); indices.push_back(bias + 1); indices.push_back(bias + 2);
			indices.push_back(bias + 0); indices.push_back(bias + 1); indices.push_back(bias + 3);
			indices.push_back(bias + 0); indices.push_back(bias + 2); indices.push_back(bias + 3);
			indices.push_back(bias + 1); indices.push_back(bias + 2); indices.push_back(bias + 3);
			bias += 4;
		}
		VkDeviceSize size = sizeof(indices[0]) * indices.size();
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(physicalDevice, device, size,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer, stagingBufferMemory);
		void* data;
		UT_CHECK_ERR(vkMapMemory(device, stagingBufferMemory, 0, size, 0, &data) == VK_SUCCESS, "failed to map memory");
		memcpy(data, indices.data(), (size_t)size);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(physicalDevice, device, size,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			indexBuffer, indexBufferMemory);
		copyBuffer(device, queue, commandPool, stagingBuffer, indexBuffer, size);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
};

class App {
public:
	App() {}
	void run() {
		initWindow();
		initVulkan();
		std::cerr << "app intialized" << std::endl;
		mainLoop();
		cleanup();
	}
private:
	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;
	VkDevice device;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	QueueFamilyIndices indices;
	SwapchainSupportDetails details;
	VkQueue uniformQueue;
	VkQueue presentQueue;
	VkSwapchainKHR swapchain;
	VkFormat swapchainImageFormat;
	VkExtent2D swapchainExtent;
	std::vector<VkImage> swapchainImages;
	std::vector<VkImageView> swapchainImageViews;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkCommandBuffer> computeCommandBuffers;
	Image2D colorImage;
	Image2D depthImage;
	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSetLayout computeDescriptorSetLayout;
	VkDescriptorPool descriptorPool;
	std::vector<VkBuffer> uniformBuffers; // for model projection matrix
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;
	std::vector<std::vector<VkDescriptorSet>> modelDescriptorSets; // [frame][modeId]
	std::vector<VkDescriptorSet> particleDescriptorSets;
	VkPipelineLayout graphicsPipelineLayout;
	VkPipeline graphicsPipeline;
	VkPipelineLayout computePipelineLayout;
	VkPipeline computePipeline;
	std::vector<VkFramebuffer> swapchainFramebuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	std::vector<VkSemaphore> computeFinishedSemaphores;
	std::vector<VkFence> computeInFlightFences;
	int currentFrame = 0;
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT; // TODO: things may be incorrecnt if msaa == 1

	std::vector<Model> models;
	ParticleSystem particleSystem;

	bool framebufferResized = false;
	const float MAX_SCROLL_SCALE = 3.0f;
	const float MIN_SCROLL_SCALE = 0.02f;
	float scrollScale = 0.25f;
	float scrollSpeed = 0.0f;

	bool isDragging = false;
	double lastMouseX = 0.0, lastMouseY = 0.0;
	float yaw = 0.0f;
	float pitch = 30.0f;
	const float mouseSensitivity = 0.1f;

	VkSampleCountFlagBits getMaxUsableSampleCount() {
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

		VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		if (counts & VK_SAMPLE_COUNT_64_BIT) return VK_SAMPLE_COUNT_64_BIT;
		if (counts & VK_SAMPLE_COUNT_32_BIT) return VK_SAMPLE_COUNT_32_BIT;
		if (counts & VK_SAMPLE_COUNT_16_BIT) return VK_SAMPLE_COUNT_16_BIT;
		if (counts & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
		if (counts & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
		if (counts & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;
		return VK_SAMPLE_COUNT_1_BIT;
	}

	// initialize the window using glfw
	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, APP_NAME.c_str(), nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		glfwSetScrollCallback(window, scrollCallback);
		glfwSetCursorPosCallback(window, mouseMoveCallback);
		glfwSetMouseButtonCallback(window, mouseButtonCallback);
	}

	// recreate swapchain upon resizing the window
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}
	
	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
		auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(window));
		app->scrollSpeed += yoffset;
	}

	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
		auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(window));

		if (button == GLFW_MOUSE_BUTTON_LEFT) {
			if (action == GLFW_PRESS) {
				glfwGetCursorPos(window, &app->lastMouseX, &app->lastMouseY);
				app->isDragging = true;
			} else if (action == GLFW_RELEASE) {
				app->isDragging = false;
			}
		}
	}

	static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
		auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(window));

		if (app->isDragging) {
			float xoffset = static_cast<float>(xpos - app->lastMouseX);
			float yoffset = static_cast<float>(app->lastMouseY - ypos);

			app->lastMouseX = xpos;
			app->lastMouseY = ypos;

			app->yaw += xoffset * app->mouseSensitivity;
			app->pitch -= yoffset * app->mouseSensitivity;

			if (app->pitch > 89.0f) app->pitch = 89.0f;
			if (app->pitch < -89.0f) app->pitch = -89.0f;
		}
	}

	// initialize vulkan
	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapchain();
		createImageViews();
		createRenderPass();
		createCommandPool();
		createCommandBuffers();
		createComputeCommandBuffers();
		createModels();
		createUniformBuffers();
		createResources();
		createDescriptorSetLayout();
		createDescriptorPool();
		createDescriptorSets();
		createGraphicsPipeline();
		createComputePipeline();
		createFramebuffers();
		createSyncObjects();
	}

	// create vulkan instance for all
	void createInstance() {
		VkApplicationInfo appInfo{
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = APP_NAME.c_str(),
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName = "No Engine",
			.engineVersion = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion = VK_API_VERSION_1_0,
		};

		if (enableValidationLayers) {
			UT_CHECK_ERR(checkRequiredLayersSupport(validationLayers) == true, "do not support validation layers");
		}
		auto extensions = getRequiredExtensions();
		VkInstanceCreateInfo createInfo{
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &appInfo,
			.enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
			.ppEnabledExtensionNames = extensions.data(),
		};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
			VkDebugUtilsMessengerCreateInfoEXT createInfo;
			populateDebugMessengerCreateInfo(createInfo);
			createInfo.pNext = (VkDebugUtilsMessengerEXT*)&createInfo;
		} else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}

		UT_CHECK_ERR(vkCreateInstance(&createInfo, nullptr, &instance) == VK_SUCCESS, "failed to create instace");
	}

	// setup the debug messenger, the debug extension must be enabled
	void setupDebugMessenger() {
		if (!enableValidationLayers) return;
		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);
		UT_CHECK_ERR(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) == VK_SUCCESS, "failed to setup debug messenger, check whether debug ext is not supported");
	}

	// create surface and bind it to the instance
	void createSurface() {
		UT_CHECK_ERR(glfwCreateWindowSurface(instance, window, nullptr, &surface) == VK_SUCCESS, "failed to create window surface");
	}

	// choose a suitable device
	void pickPhysicalDevice() {
		uint32_t cnt = 0;
		vkEnumeratePhysicalDevices(instance, &cnt, nullptr);
		UT_CHECK_ERR(cnt != 0, "failed to find GPUs with Vulkan support");
		
		std::vector<VkPhysicalDevice> devices(cnt);
		vkEnumeratePhysicalDevices(instance, &cnt, devices.data());
		for (const auto& device : devices) {
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(device, &deviceProperties);
			indices = findQueueFamilyProperties(device, surface);
			details = querySwapchainSupport(device, surface);
			if (isDeviceSuitable(device, surface)) {
				physicalDevice = device;
				msaaSamples = getMaxUsableSampleCount();
				break;
			}
		}
		UT_CHECK_ERR(physicalDevice != VK_NULL_HANDLE, "failed to find a suitabale GPU");
	}

	// create logical device with queues and enable device features
	void createLogicalDevice() {
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.uniformFamily.value(), indices.presentFamily.value() };
		float queuePriority = 1.0f;

		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = queueFamily,
				.queueCount = 1,
				.pQueuePriorities = &queuePriority,
			};
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{
			.samplerAnisotropy = VK_TRUE,
		};
		VkDeviceCreateInfo createInfo{
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
			.pQueueCreateInfos = queueCreateInfos.data(),
			.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
			.ppEnabledExtensionNames = deviceExtensions.data(),
			.pEnabledFeatures = &deviceFeatures,
		};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}
		UT_CHECK_ERR(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) == VK_SUCCESS, "failed to create logical device");

		// fetch queues
		vkGetDeviceQueue(device, indices.uniformFamily.value(), 0, &uniformQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	// swapchain
	void createSwapchain() {
		// details should be recreated too
		details = querySwapchainSupport(physicalDevice, surface);
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(details.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(details.presentModes);
		VkExtent2D extent = chooseSwapExtent(window, details.capabilities);
		uint32_t imageCount = details.capabilities.minImageCount + 1;
		if (details.capabilities.maxImageCount > 0 && imageCount > details.capabilities.maxImageCount) {
			imageCount = details.capabilities.maxImageCount;
		}
		VkSwapchainCreateInfoKHR createInfo{
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = surface,
			.minImageCount = imageCount,
			.imageFormat = surfaceFormat.format,
			.imageColorSpace = surfaceFormat.colorSpace,
			.imageExtent = extent,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.preTransform = details.capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = presentMode,
			.clipped = VK_TRUE,
			.oldSwapchain = VK_NULL_HANDLE,
		};

		uint32_t queueFamilyIndices[] = { indices.uniformFamily.value(), indices.presentFamily.value() };
		if (indices.uniformFamily != indices.presentFamily) { // TODO: fix WAW problem
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		} else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}
		UT_CHECK_ERR(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) == VK_SUCCESS, "failed to create swapchain");
		vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
		swapchainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
		swapchainImageFormat = surfaceFormat.format;
		swapchainExtent = extent;
	}

	// image views to swapchain
	void createImageViews() {
		swapchainImageViews.resize(swapchainImages.size());
		for (int i = 0; i < swapchainImages.size(); ++i) {
			swapchainImageViews[i] = createImageView(device, swapchainImages[i], swapchainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		}
	}

	// command pool
	void createCommandPool() {
		VkCommandPoolCreateInfo poolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = indices.uniformFamily.value(),
		};
		UT_CHECK_ERR(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) == VK_SUCCESS, "failed to create command pool");
	}

	// command buffer
	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = (uint32_t)commandBuffers.size(),
		};
		UT_CHECK_ERR(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) == VK_SUCCESS, "failed to allocate command buffers");
	}

	// compute command buffer
	void createComputeCommandBuffers() {
		computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = (uint32_t)computeCommandBuffers.size(),
		};
		UT_CHECK_ERR(vkAllocateCommandBuffers(device, &allocInfo, computeCommandBuffers.data()) == VK_SUCCESS, "failed to allocate compute command buffers");
	}

	// render pass
	void createRenderPass() {
		VkAttachmentDescription colorAttachment{
			.format = swapchainImageFormat,
			.samples = msaaSamples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
		VkAttachmentDescription depthAttachment{
			.format = findDepthFormat(physicalDevice),
			.samples = msaaSamples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};
		VkAttachmentDescription colorAttachmentResolve{
			.format = swapchainImageFormat,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		};
		VkAttachmentReference colorAttachmentRef{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
		VkAttachmentReference depthAttachmentRef{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};
		VkAttachmentReference colorAttachmentResolveRef{
			.attachment = 2,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachmentRef,
			.pResolveAttachments = &colorAttachmentResolveRef,
			.pDepthStencilAttachment = &depthAttachmentRef,
		};
		std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };

		VkSubpassDependency dependency{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		};

		VkRenderPassCreateInfo renderPassInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = static_cast<uint32_t>(attachments.size()),
			.pAttachments = attachments.data(),
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = 1,
			.pDependencies = &dependency,
		};

		UT_CHECK_ERR(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) == VK_SUCCESS, "failed to create render pass");
	}

	// descriptor, only related to shader
	void createDescriptorSetLayout() {
		VkDescriptorSetLayoutBinding uboLayoutBinding{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
			.pImmutableSamplers = nullptr,
		};

		VkDescriptorSetLayoutBinding samplerLayoutBinding{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			.pImmutableSamplers = nullptr,
		};

		VkDescriptorSetLayoutBinding SSBOInLayoutBinding{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.pImmutableSamplers = nullptr,
		};

		VkDescriptorSetLayoutBinding SSBOOutLayoutBinding{
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.pImmutableSamplers = nullptr,
		};
		std::array<VkDescriptorSetLayoutBinding, 4> bindings = { uboLayoutBinding, samplerLayoutBinding, SSBOInLayoutBinding, SSBOOutLayoutBinding};

		VkDescriptorSetLayoutCreateInfo layoutInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = static_cast<uint32_t>(bindings.size()),
			.pBindings = bindings.data(),
		};

		UT_CHECK_ERR(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) == VK_SUCCESS, "failed to create descriptor set layout");
	}

	// color
	void createResources() {
		VkFormat colorFormat = swapchainImageFormat;
		uint32_t width = swapchainExtent.width;
		uint32_t height = swapchainExtent.height;
		colorImage.create(physicalDevice, device, width, height, 1, colorFormat,
			msaaSamples,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_IMAGE_ASPECT_COLOR_BIT);
		auto depthFormat = findDepthFormat(physicalDevice);
		depthImage.create(physicalDevice, device, width, height, 1, depthFormat,
			msaaSamples,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT);
	}

	// uniform buffers
	void createUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(MVP) * (models.size() + 1);
		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			createBuffer(physicalDevice, device, bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				uniformBuffers[i], uniformBuffersMemory[i]);
			vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}
	}

	void createModels() {
		models.push_back(Model("./models/ball.obj", "./textures/sun.png", physicalDevice, device, uniformQueue, commandPool,
			1.0f, 0.0f, 1.0f, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f)));
		models.push_back(Model("./models/ball.obj", "./textures/earth.png", physicalDevice, device, uniformQueue, commandPool,
			0.3f, 1.0f, 1.0f, glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f)));
		particleSystem.create(physicalDevice, device, uniformQueue, commandPool, 512, "./textures/particle.png");
	}

	void createDescriptorPool() {
		std::array<VkDescriptorPoolSize, 3> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * (models.size() + 5));
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * (models.size() + 1));
		poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 2);
		VkDescriptorPoolCreateInfo poolInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * (models.size() + 1)), // models + particle system
			.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
			.pPoolSizes = poolSizes.data(),
		};
		UT_CHECK_ERR(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) == VK_SUCCESS, "failed to create descriptor pool");
	}

	void createDescriptorSets() {
		modelDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		particleDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

		std::vector<VkDescriptorSetLayout> layoutsParticle(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		VkDescriptorSetAllocateInfo particleAllocInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptorPool,
			.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
			.pSetLayouts = layoutsParticle.data(),
		};
		UT_CHECK_ERR(vkAllocateDescriptorSets(device, &particleAllocInfo, particleDescriptorSets.data()) == VK_SUCCESS, "failed to allocate descriptor sets");
		
		std::vector<VkDescriptorSetLayout> layouts(models.size(), descriptorSetLayout);
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			std::vector<VkDescriptorSet> sets;
			sets.resize(models.size());
			VkDescriptorSetAllocateInfo allocInfo{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptorPool,
				.descriptorSetCount = static_cast<uint32_t>(models.size()),
				.pSetLayouts = layouts.data(),
			};
			auto res = vkAllocateDescriptorSets(device, &allocInfo, sets.data());
			UT_CHECK_ERR(res == VK_SUCCESS, "failed to allocate descriptor sets");
			// descriptors for models
			for (int modelId = 0; modelId < models.size(); ++modelId) {
				VkDescriptorBufferInfo bufferInfo{
					.buffer = uniformBuffers[i],
					.offset = sizeof(MVP) * modelId, // use the j-th MVP object
					.range = sizeof(MVP),
				};
				std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = sets[modelId];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[0].pImageInfo = nullptr;
				descriptorWrites[0].pBufferInfo = &bufferInfo;
				descriptorWrites[0].pTexelBufferView = nullptr;

				auto info = models[modelId].getDescriptorImageInfo();
				descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[1].dstSet = sets[modelId];
				descriptorWrites[1].dstBinding = 1;
				descriptorWrites[1].dstArrayElement = 0;
				descriptorWrites[1].descriptorCount = 1;
				descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[1].pImageInfo = &info;
				descriptorWrites[1].pBufferInfo = nullptr;
				descriptorWrites[1].pTexelBufferView = nullptr;
				vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
			modelDescriptorSets[i] = sets;

			// descriptors for particle system
			std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
			
			// MVP for particle system
			VkDescriptorBufferInfo bufferInfo{
				.buffer = uniformBuffers[i],
				.offset = sizeof(MVP) * models.size(),
				.range = sizeof(MVP),
			};
			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = particleDescriptorSets[i];
			descriptorWrites[0].dstBinding = 0; // uniform buffer
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].pImageInfo = nullptr;
			descriptorWrites[0].pBufferInfo = &bufferInfo;
			descriptorWrites[0].pTexelBufferView = nullptr;

			auto info = particleSystem.getDescriptorImageInfo();
			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = particleDescriptorSets[i];
			descriptorWrites[1].dstBinding = 1; // texture sampler
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].pImageInfo = &info;
			descriptorWrites[1].pBufferInfo = nullptr;
			descriptorWrites[1].pTexelBufferView = nullptr;

			auto lastInfo = particleSystem.getDescriptorObjectBufferInfo(i);
			descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[2].dstSet = particleDescriptorSets[i];
			descriptorWrites[2].dstBinding = 2; // object input
			descriptorWrites[2].dstArrayElement = 0;
			descriptorWrites[2].descriptorCount = 1;
			descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[2].pImageInfo = nullptr;
			descriptorWrites[2].pBufferInfo = &lastInfo;
			descriptorWrites[2].pTexelBufferView = nullptr;

			auto currentInfo = particleSystem.getDescriptorVertexBufferInfo(i);
			descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[3].dstSet = particleDescriptorSets[i];
			descriptorWrites[3].dstBinding = 3; // vertex output
			descriptorWrites[3].dstArrayElement = 0;
			descriptorWrites[3].descriptorCount = 1;
			descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[3].pImageInfo = nullptr;
			descriptorWrites[3].pBufferInfo = &currentInfo;
			descriptorWrites[3].pTexelBufferView = nullptr;
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("./shaders/vert.spv");
		auto fragShaderCode = readFile("./shaders/frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertShaderModule,
			.pName = "main",
		};

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragShaderModule,
			.pName = "main",
		};

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		auto bindingDescription = Vertex::getBindingDescription();
		auto attrbuteDescriptions = Vertex::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrbuteDescriptions.size()),
			.pVertexAttributeDescriptions = attrbuteDescriptions.data(),
		};

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = (float)swapchainExtent.width,
			.height = (float)swapchainExtent.height,
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = swapchainExtent,
		};

		VkPipelineViewportStateCreateInfo viewportState{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &viewport,
			.scissorCount = 1,
			.pScissors = &scissor,
		};

		
		VkPipelineRasterizationStateCreateInfo rasterizer{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = VK_FALSE,
			.depthBiasConstantFactor = 0.0f,
			.depthBiasClamp = 0.0f,
			.depthBiasSlopeFactor = 0.0f,
			.lineWidth = 1.0f,
		};

		VkPipelineMultisampleStateCreateInfo multisampling{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = msaaSamples,
			.sampleShadingEnable = VK_FALSE,
			.minSampleShading = 1.0f,
			.pSampleMask = nullptr,
			.alphaToCoverageEnable = VK_FALSE,
			.alphaToOneEnable = VK_FALSE,
		};

		VkPipelineDepthStencilStateCreateInfo depthStencil{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
			.minDepthBounds = 0.0f,
			.maxDepthBounds = 1.0f,
		};

		VkPipelineColorBlendAttachmentState colorBlendAttachment{
			.blendEnable = VK_FALSE,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};

		VkPipelineColorBlendStateCreateInfo colorBlending{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = VK_FALSE,
			.logicOp = VK_LOGIC_OP_COPY,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment,
		};
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptorSetLayout,
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = nullptr,
		};
		UT_CHECK_ERR(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &graphicsPipelineLayout) == VK_SUCCESS, "failed to create pipeline layout");

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
			.pDynamicStates = dynamicStates.data(),
		};

		VkGraphicsPipelineCreateInfo pipelineInfo{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = shaderStages,
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = graphicsPipelineLayout,
			.renderPass = renderPass,
			.subpass = 0,
			.basePipelineHandle = VK_NULL_HANDLE,
			.basePipelineIndex = -1,
		};
		UT_CHECK_ERR(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) == VK_SUCCESS, "failed to create graphics pipeline");

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	void createComputePipeline() {
		auto computeShaderCode = readFile("shaders/compute.spv");

		VkShaderModule computeShaderModule = createShaderModule(device, computeShaderCode);

		VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		computeShaderStageInfo.module = computeShaderModule;
		computeShaderStageInfo.pName = "main";

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptorSetLayout,
		};

		UT_CHECK_ERR(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) == VK_SUCCESS, "failed to create compute pipeline");

		VkComputePipelineCreateInfo pipelineInfo{
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage = computeShaderStageInfo,
			.layout = computePipelineLayout,
		};

		UT_CHECK_ERR(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) == VK_SUCCESS, "failed to create compute pipeline");

		vkDestroyShaderModule(device, computeShaderModule, nullptr);
	}

	void createFramebuffers() {
		swapchainFramebuffers.resize(swapchainImageViews.size());
		for (int i = 0; i < swapchainImageViews.size(); ++i) {
			std::array<VkImageView, 3> attachments = { colorImage.imageView, depthImage.imageView, swapchainImageViews[i] };
			VkFramebufferCreateInfo framebufferInfo{
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass = renderPass,
				.attachmentCount = static_cast<uint32_t>(attachments.size()),
				.pAttachments = attachments.data(),
				.width = swapchainExtent.width,
				.height = swapchainExtent.height,
				.layers = 1,
			};
			UT_CHECK_ERR(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapchainFramebuffers[i]) == VK_SUCCESS, "failed to create framebuffer");
		}
	}

	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		VkSemaphoreCreateInfo semaphoreInfo{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};

		VkFenceCreateInfo fenceInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			UT_CHECK_ERR(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) == VK_SUCCESS, "failed to create semaphores");
			UT_CHECK_ERR(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) == VK_SUCCESS, "failed to create semaphores");
			UT_CHECK_ERR(vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) == VK_SUCCESS, "failed to create fence");
			UT_CHECK_ERR(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) == VK_SUCCESS, "failed to create semaphores");
			UT_CHECK_ERR(vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[i]) == VK_SUCCESS, "failed to create fence");
		}
	}

	void cleanupSwapchain() {
		colorImage.destroy(device);
		depthImage.destroy(device);
		for (auto framebuffer : swapchainFramebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		for (auto imageView : swapchainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapchain, nullptr);
	}

	void mainLoop() {
		int width, height;
		glfwGetWindowSize(window, &width, &height);
		auto lastTime = std::chrono::high_resolution_clock::now();
		auto firstTime = std::chrono::high_resolution_clock::now();
		int frames = 0;
		while (!glfwWindowShouldClose(window)) {
			auto currentTime = std::chrono::high_resolution_clock::now();
			glfwGetWindowSize(window, &width, &height);
			while (width == 0 || height == 0) {
				glfwWaitEvents();
				glfwGetWindowSize(window, &width, &height);
			}
			frames += 1;
			auto fpsDeltaTime = std::chrono::duration<double>(currentTime - lastTime).count();
			auto deltaTime = std::chrono::duration<double>(currentTime - firstTime).count();
			drawFrame(deltaTime);
			glfwPollEvents();
			if (fpsDeltaTime > 0.5) {
				lastTime = currentTime;
				double fps = frames / fpsDeltaTime;
				frames = 0;
				std::string title = APP_NAME + "- FPS: " + std::to_string((int)fps);
				glfwSetWindowTitle(window, title.c_str());
			}

			const float c = 1.0f;
			const float t = 100.0f;
			const float d = 0.1f;
			float resist = std::max(c / (MAX_SCROLL_SCALE - scrollScale), c / (scrollScale - MIN_SCROLL_SCALE));
			scrollSpeed *= 0.98;
			float updateSpeed = std::clamp(scrollSpeed, -d, d);
			updateSpeed = std::min(updateSpeed, MAX_SCROLL_SCALE - scrollScale);
			updateSpeed = std::max(updateSpeed, MIN_SCROLL_SCALE - scrollScale);
			scrollScale = scrollScale + updateSpeed / std::max(t, resist);
		}
		vkDeviceWaitIdle(device);
	}

	void cleanup() {
		cleanupSwapchain();

		vkDestroyPipeline(device, computePipeline, nullptr);
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
		vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);
		for (auto& model : models) {
			model.destroy(device);
		}
		particleSystem.destroy(device);
		
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
			vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, computeInFlightFences[i], nullptr);
		}

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		vkDestroyDevice(device, nullptr);
		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void recreateSwapchain() {
		vkDeviceWaitIdle(device);

		cleanupSwapchain();

		createSwapchain();
		createImageViews();
		createResources();
		createFramebuffers();
	}

	void updateUniformBuffer(uint32_t index, float deltaTime) {
		static auto startTime = std::chrono::high_resolution_clock::now();
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		//auto view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

		float radius = 3.0;
		glm::vec3 cameraPos;
		cameraPos.x = radius * cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		cameraPos.y = radius * sin(glm::radians(yaw)) * cos(glm::radians(pitch));
		cameraPos.z = radius * sin(glm::radians(pitch));

		glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);
		glm::mat4 view = glm::lookAt(cameraPos, lookAt, up);

		auto proj = glm::perspective(glm::radians(45.0f), swapchainExtent.width / (float)swapchainExtent.height, 0.1f, 10.0f);

		auto scaleM = glm::scale(glm::mat4(1.0f), glm::vec3(scrollScale, scrollScale, scrollScale));

		std::vector<MVP> ubos;
		for (int i = 0; i < models.size(); ++i) {
			auto model = scaleM;
			model = glm::translate(model, models[i].pivot);
			model = glm::rotate(model, time * glm::radians(90.0f) * models[i].speed, glm::vec3(0.0f, 0.0f, 1.0f)); // then apply rotation
			model = glm::translate(model, models[i].origin - models[i].pivot); // then move to origin - pivot
			model = glm::rotate(model, time * glm::radians(90.0f) * models[i].speedSelf, glm::vec3(0.0f, 0.0f, 1.0f)); // then self rotatation
			model = glm::scale(model, glm::vec3(models[i].sizeScale, models[i].sizeScale, models[i].sizeScale)); // scale first
			MVP ubo{
				.model = model,
				.view = view,
				.proj = proj,
				.deltaTime = deltaTime,
			};
			ubo.proj[1][1] *= -1;
			ubos.push_back(ubo);
		}

		// particle system
		auto model = scaleM;
		MVP ubo{
			.model = model,
			.view = view,
			.proj = proj,
			.deltaTime = deltaTime,
		};
		ubo.proj[1][1] *= -1;
		ubos.push_back(ubo);
		memcpy(uniformBuffersMapped[index], ubos.data(), sizeof(MVP) * (models.size() + 1));
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t frameIndex) {

		VkCommandBufferBeginInfo beginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = 0,
			.pInheritanceInfo = nullptr,
		};
		// start recording
		UT_CHECK_ERR(vkBeginCommandBuffer(commandBuffer, &beginInfo) == VK_SUCCESS, "failed to begin recording command buffer");
		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };
		VkRenderPassBeginInfo renderPassInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = renderPass,
			.framebuffer = swapchainFramebuffers[imageIndex],
			.renderArea = {
				.offset = {0, 0},
				.extent = swapchainExtent,
			},
			.clearValueCount = static_cast<uint32_t>(clearValues.size()),
			.pClearValues = clearValues.data(),
		};

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
		// chage the dynamic states
		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(swapchainExtent.width),
			.height = static_cast<float>(swapchainExtent.height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = swapchainExtent,
		};
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		for (int i = 0; i < models.size(); ++i) {

			VkBuffer vertexBuffers[] = { models[i].vertexBuffer};
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(commandBuffer, models[i].indexBuffer, 0, VK_INDEX_TYPE_UINT32);

			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &modelDescriptorSets[frameIndex][i], 0, nullptr);
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(models[i].indices.size()), 1, 0, 0, 0);
		}

		// draw particle system
		VkBuffer vertexBuffers[] = { particleSystem.vertexBuffer[frameIndex] };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, particleSystem.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &particleDescriptorSets[frameIndex], 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(particleSystem.indices.size()), 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		UT_CHECK_ERR(vkEndCommandBuffer(commandBuffer) == VK_SUCCESS, "failed to record command buffer");
	}

	void recordComputeCommandBuffer(VkCommandBuffer commandBuffer, uint32_t frameIndex) {
		VkCommandBufferBeginInfo beginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		};
		UT_CHECK_ERR(vkBeginCommandBuffer(commandBuffer, &beginInfo) == VK_SUCCESS, "failed to begain recording compute buffer");
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &particleDescriptorSets[frameIndex], 0, nullptr);
		vkCmdDispatch(commandBuffer, (particleSystem.particles.size() + 255) / 256, 1, 1);

		UT_CHECK_ERR(vkEndCommandBuffer(commandBuffer) == VK_SUCCESS, "failed to record command buffer");
	}

	void drawFrame(double deltaTime) {
		vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &computeInFlightFences[currentFrame]);
		vkResetCommandBuffer(computeCommandBuffers[currentFrame], 0);
		recordComputeCommandBuffer(computeCommandBuffers[currentFrame], currentFrame);
		VkSubmitInfo submitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &computeCommandBuffers[currentFrame],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &computeFinishedSemaphores[currentFrame],
		};

		UT_CHECK_ERR(vkQueueSubmit(uniformQueue, 1, &submitInfo, computeInFlightFences[currentFrame]) == VK_SUCCESS, "failed to submit compute command buffer");

		// wait until graphics frame is done
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapchain();
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swapchain image");
		}
		updateUniformBuffer(currentFrame, deltaTime);
		// reset the fence
		vkResetFences(device, 1, &inFlightFences[currentFrame]);
		//// update uniform buffers like camera, model transformation, ...
		//updateUniformBuffer(currentFrame, deltaTime);
		// reset the command buffer
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		// record commands
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex, currentFrame);

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		VkSemaphore waitSemaphores[] = { computeFinishedSemaphores[currentFrame], imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; // wait for compute shader to finish updating vertices
		submitInfo = VkSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 2,
			.pWaitSemaphores = waitSemaphores,
			.pWaitDstStageMask = waitStages,
			.commandBufferCount = 1,
			.pCommandBuffers = &commandBuffers[currentFrame],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = signalSemaphores,
		};

		VkSwapchainKHR swapchains[] = { swapchain };
		UT_CHECK_ERR(vkQueueSubmit(uniformQueue, 1, &submitInfo, inFlightFences[currentFrame]) == VK_SUCCESS, "failed to draw command buffer");

		VkPresentInfoKHR presentInfo{
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = signalSemaphores,
			.swapchainCount = 1,
			.pSwapchains = swapchains,
			.pImageIndices = &imageIndex,
			.pResults = nullptr,
		};
		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapchain();
		} else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present the swap chain image");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}
};


// get glfw & validation extensions
static std::vector<const char*> getRequiredExtensions() {
	uint32_t cnt = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&cnt);
	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + cnt);
	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	return extensions;
}

// chech if the device supports swapchain, extensions and features
static bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR& surface) {
	QueueFamilyIndices indices = findQueueFamilyProperties(device, surface);
	bool extensionsSupported = checkRequiredExtensionsSupport(device, deviceExtensions);
	bool swapChainAdequate = false;
	if (extensionsSupported) { // swapchain is included
		SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device, surface);
		swapChainAdequate = !swapchainSupport.formats.empty() && !swapchainSupport.presentModes.empty();
	}
	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

	return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

static SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR& surface) {
	SwapchainSupportDetails details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
	uint32_t cnt;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &cnt, nullptr);
	if (cnt != 0) {
		details.formats.resize(cnt);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &cnt, details.formats.data());
	}
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &cnt, nullptr);
	if (cnt != 0) {
		details.presentModes.resize(cnt);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &cnt, details.presentModes.data());
	}
	return details;
}

static QueueFamilyIndices findQueueFamilyProperties(VkPhysicalDevice device, VkSurfaceKHR& surface) {
	QueueFamilyIndices indices;
	uint32_t cnt;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &cnt, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(cnt);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &cnt, queueFamilies.data());
	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
			//if (!indices.uniformFamily.has_value()) {
				indices.uniformFamily = i;
			//}
		}
		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
		if (presentSupport) {
			indices.presentFamily = i;
		}
		if (indices.isComplete()) {
			break;
		}
		++i;
	}
	std::cout << indices.presentFamily.value() << " " << indices.uniformFamily.value() << std::endl;
	return indices;
}

static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
	for (const auto& format : availableFormats) {
		if (format.format == VK_FORMAT_R8G8B8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return format;
		}
	}
	return availableFormats[0];
}

static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
	for (const auto& mode : availablePresentModes) {
		if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
			return mode;
		}
	}
	return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities) {
	if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
		return capabilities.currentExtent;
	}
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	VkExtent2D actualExtent = {
		static_cast<uint32_t>(width),
		static_cast<uint32_t>(height),
	};
	actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
	actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
	return actualExtent;
}

// create a 2D image view
static VkImageView createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLeveles) {
	VkImageViewCreateInfo viewInfo{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {
			.aspectMask = aspectFlags,
			.baseMipLevel = 0,
			.levelCount = mipLeveles,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	VkImageView imageView;
	UT_CHECK_ERR(vkCreateImageView(device, &viewInfo, nullptr, &imageView) == VK_SUCCESS, "failed to create texture image view");
	return imageView;
}

static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}
	throw std::runtime_error("failed to find suitable memory type");
}

static VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
	for (VkFormat format : candidates) {
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
			return format;
		} else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
			return format;
		} else {
			throw std::runtime_error("not supported tiling format");
		}
	}
	throw std::runtime_error("failed to find supported format");
}

static bool hasStencilComponent(VkFormat format) {
	return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

static VkFormat findDepthFormat(VkPhysicalDevice physicalDevice) {
	return findSupportedFormat(physicalDevice,
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

static void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
	VkBufferCreateInfo bufferInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	UT_CHECK_ERR(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) == VK_SUCCESS, "failed to create buffer");

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties),
	};

	UT_CHECK_ERR(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) == VK_SUCCESS, "failed to allocate buffer memory");
	vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

static VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool) {
	VkCommandBufferAllocateInfo allocInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = commandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
	VkCommandBufferBeginInfo beginInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(commandBuffer, &beginInfo);
	return commandBuffer;
}

static void endSingleTimeCommands(VkDevice device, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer) {
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &commandBuffer,
	};

	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

static void copyBufferToImage(VkDevice device, VkQueue queue, VkCommandPool commandPool, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
	VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
	VkBufferImageCopy region{
		.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.imageOffset = {0, 0, 0,},
		.imageExtent = {
			width,
			height,
			1
		},
	};
	vkCmdCopyBufferToImage(
		commandBuffer,
		buffer,
		image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&region
	);
	endSingleTimeCommands(device, queue, commandPool, commandBuffer);
}

static void generateMipmaps(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, VkCommandPool commandPool, VkImage image, VkFormat imageFormat, int32_t width, int32_t height, uint32_t mipLevels) {
	VkFormatProperties formatProperties;
	vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);
	// TODO: add feature check
	if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
		throw std::runtime_error("texture image format does not support linear blitting");
	}
	VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
	VkImageMemoryBarrier barrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	int32_t mipWidth = width;
	int32_t mipHeight = height;
	for (uint32_t i = 1; i < mipLevels; ++i) {
		barrier.subresourceRange.baseMipLevel = i - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);
		VkImageBlit blit{
			.srcSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = i - 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.srcOffsets = {
				{0, 0, 0,},
				{mipWidth, mipHeight, 1},
			},
			.dstSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = i,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.dstOffsets = {
				{0, 0, 0},
				{mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 },
			},
		};
		vkCmdBlitImage(
			commandBuffer,
			image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &blit,
			VK_FILTER_LINEAR
		);
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);
		if (mipWidth > 1) mipWidth /= 2;
		if (mipHeight > 1) mipHeight /= 2;
	}
	barrier.subresourceRange.baseMipLevel = mipLevels - 1;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);
	endSingleTimeCommands(device, queue, commandPool, commandBuffer);
}

static void transitionImageLayout(VkDevice device, VkQueue queue, VkCommandPool commandPool, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
	VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
	VkImageMemoryBarrier barrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = 0,
		.dstAccessMask = 0,
		.oldLayout = oldLayout,
		.newLayout = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = mipLevels,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		if (hasStencilComponent(format)) {
			barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}
	}
	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	} else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	} else {
		throw std::runtime_error("unsupporte layout transition");
	}
	vkCmdPipelineBarrier(
		commandBuffer,
		sourceStage, destinationStage,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);
	endSingleTimeCommands(device, queue, commandPool, commandBuffer);
}

static void createImage(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format,
	VkSampleCountFlagBits numSamples, VkImageTiling tiling, VkImageUsageFlags usage,
	VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {

	VkImageCreateInfo imageInfo{
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.flags = 0,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = {
			.width = width,
			.height = height,
			.depth = 1,
		},
		.mipLevels = mipLevels,
		.arrayLayers = 1,
		.samples = numSamples,
		.tiling = tiling,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	UT_CHECK_ERR(vkCreateImage(device, &imageInfo, nullptr, &image) == VK_SUCCESS, "failed to create image");

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(device, image, &memRequirements);
	VkMemoryAllocateInfo allocInfo{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties),
	};
	UT_CHECK_ERR(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) == VK_SUCCESS, "failed to allocate image memory");
	vkBindImageMemory(device, image, imageMemory, 0);
}

static void copyBuffer(VkDevice device, VkQueue queue, VkCommandPool commandPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
	VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
	VkBufferCopy copyRegion{
		.srcOffset = 0,
		.dstOffset = 0,
		.size = size,
	};
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
	endSingleTimeCommands(device, queue, commandPool, commandBuffer);
}

static std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open file");
	}
	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
	VkShaderModuleCreateInfo createInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data())
	};
	VkShaderModule shaderModule;
	UT_CHECK_ERR(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) == VK_SUCCESS, "failed to create shader module");
	return shaderModule;
}

int main() {
	App app;
	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
